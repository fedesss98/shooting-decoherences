const I2 = [1.0+0.0im 0.0; 0.0 1.0]
const Z  = [1.0+0.0im 0.0; 0.0 -1.0]
const X  = [0.0im 1.0; 1.0 0.0]


"""
    kraus_depolarizing(p::Float64)

Returns Kraus operators for the depolarizing channel.
See for example
    https://www.preskill.caltech.edu/ph219/chap3_15.pdf#page=24.11
"""
function get_depolarizing_operators(gamma, t)
    p = 1 - exp(-gamma * t)
    # Probability of no error
    p_i = 1 - p
    # Probability of each specific Pauli error (X, Y, Z)
    p_err = p / 3.0
    
    K0 = sqrt(p_i) * [1.0 0.0; 0.0 1.0]
    K1 = sqrt(p_err) * [0.0 1.0; 1.0 0.0]        # X
    K2 = sqrt(p_err) * [0.0 -im; im 0.0]         # Y
    K3 = sqrt(p_err) * [1.0 0.0; 0.0 -1.0]       # Z
    
    return [K0, K1, K2, K3]
end


"""
    get_amplitudedamping_operators(gamma, t)
See for example
    https://www.preskill.caltech.edu/ph219/chap3_15.pdf#page=24.11
    https://docs.pennylane.ai/en/stable/code/api/pennylane.AmplitudeDamping.html
"""
function get_amplitudedamping_operators(gamma, t)
    e = exp(-gamma*t)
    k1 = [
        1 0;
        0 sqrt(e)
    ]
    k2 = [
        0 sqrt(1-e)
        0 0
    ]
    return [k1, k2]
end

"""
    get_phasedamping_operators(gamma, t)
See for example
    https://docs.pennylane.ai/en/stable/code/api/pennylane.PhaseDamping.html
"""
function get_phasedamping_operators(gamma, t)
    e = exp(-gamma*t)
    k1 = [
        1 0;
        0 sqrt(e)
    ]
    k2 = [
        0 0
        0 sqrt(1-e)
    ]
    return [k1, k2]
end

"""
    get_dephasing_operators(gamma, t)
See for example
    https://www.preskill.caltech.edu/ph219/chap3_15.pdf#page=24.11
"""
function get_dephasing_operators(gamma, t)
    p = (1.0 - exp(-gamma*t)) / 2.0

    k1 = sqrt(1.0 - p) * I2
    k2 = sqrt(p) * Z
    
    return [k1, k2]
end

"""
    get_bitflip_operators(gamma, t)
See for example
    https://docs.pennylane.ai/en/stable/code/api/pennylane.BitFlip.html
"""
function get_bitflip_operators(gamma, t)
    p = (1.0 - exp(-gamma*t)) / 2.0

    k1 = sqrt(1.0 - p) * I2
    k2 = sqrt(p) * X
    
    return [k1, k2]
end

"""
    apply_channel(kraus, ρ)
"""
function apply_channel(kraus, ρ)
    return sum([k*ρ*k' for k in kraus])
end

"""
    apply_channel(kraus, ρ, n_qubits)
Applies the amplitude damping channel for all n qubits in the system sequentially
"""
function apply_channel(kraus, ρ, n_qubits::Int)
    ρi = copy(ρ)
    
    if n_qubits == 1
        return apply_channel(kraus, ρ)
    end

    for qubit in 1:n_qubits
        ρf = zeros(ComplexF64, size(ρ))
        for k in kraus
            # Extend the Kraus operator
            op_list = [i == qubit ? k : I2 for i in 1:n_qubits]
            k_full = foldl(kron, op_list)

            ρf += k_full * ρi * k_full'
        end
        # Update state
        ρi = ρf
    end

    return ρi
end


"""
    partial_recovery_function(kraus, σ, n_qubits)
"""
function partial_recovery_function(kraus, σ, n_qubits)
    # Define the evolution map and its adjoint
    map(x) = apply_channel(kraus, x, n_qubits)

    sqrt_map = matrix_power_pseudo(map(σ), -0.5)
    sqrt_state = matrix_power_pseudo(σ, 0.5)
    kraus_dagger = [k' for k in kraus]
    return ρ -> begin
        map_adj(x) = apply_channel(kraus_dagger, x, n_qubits)
        inner = sqrt_map * ρ * sqrt_map
        recovered = sqrt_state * map_adj(inner) * sqrt_state
        return recovered
    end
end

"""
    recovery_map(kraus, σ, ρ)
Returns the recovery map as a function of the new state to be recovered only
"""
function recovery_map(kraus, σ, ρ)
    map(x) = sum([k*x*k' for k in kraus])
    map_adj(x) = sum([k'*x*k for k in kraus])

    inner = matrix_power_pseudo(map(σ), -0.5) * ρ * matrix_power_pseudo(map(σ), -0.5)
    recovered = matrix_power_pseudo(σ, 0.5) * map_adj(inner) * matrix_power_pseudo(σ, 0.5)
end

"""
    recovery_map(kraus, σ, ρ, n_qubits)
Applies the recovery map for all n qubits in the system sequentially
"""
function recovery_map(kraus, σ, ρ, n_qubits)
    # Define the evolution map and its adjoint
    kraus_dagger = [k' for k in kraus]
    map(ρ) = apply_channel(kraus, ρ, n_qubits)
    map_adj(ρ) = apply_channel(kraus_dagger, ρ, n_qubits)

    inner = matrix_power_pseudo(map(σ), -0.5) * ρ * matrix_power_pseudo(map(σ), -0.5)
    recovered = matrix_power_pseudo(σ, 0.5) * map_adj(inner) * matrix_power_pseudo(σ, 0.5)

    return recovered
end


"""
    get_kraus_from_map(E, d; tol=1e-10)

Given a quantum channel `E` specified as a function on density matrices
(`ρ::AbstractMatrix{<:Complex}`), return a vector of Kraus operators `K_k`
as `Matrix{ComplexF64}`, such that

    E(ρ) ≈ sum(K -> K * ρ * K', Ks)

for all ρ in the d-dimensional system.
"""
function get_kraus_from_map(E::Function; d::Int, tol=1e-10)
    # Build Choi matrix J of size d^2 × d^2
    J = zeros(ComplexF64, d^2, d^2)

    # Basis operators |m><n|
    for m in 1:d, n in 1:d
        ρ_mn = zeros(ComplexF64, d, d)
        ρ_mn[m, n] = 1.0
        F_mn = E(ρ_mn)  # d×d matrix

        # Fill the corresponding block in J
        # row index: (m-1)*d + p
        # col index: (n-1)*d + q
        @inbounds for p in 1:d, q in 1:d
            row = (m-1)*d + p
            col = (n-1)*d + q
            J[row, col] = F_mn[p, q]
        end
    end

    # Symmetrize to remove small numerical non-Hermiticity
    J = (J + J') / 2

    # Eigen-decomposition: J = V * Diagonal(λ) * V'
    evals, evecs = eigen(Hermitian(J))

    kraus = Matrix{ComplexF64}[]
    for k in eachindex(evals)
        λ = evals[k]
        if λ > tol
            v = evecs[:, k]              # length d^2
            K = reshape(v, d, d)         # d × d matrix
            push!(kraus, sqrt(λ) * K)
        end
    end

    return kraus
end
