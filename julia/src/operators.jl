const I2 = [1.0+0.0im 0.0; 0.0 1.0]
const Z  = [1.0+0.0im 0.0; 0.0 -1.0]
const X  = [0.0im 1.0; 1.0 0.0]


"""
    density_matrix(a)

a ∈ [-1, 1]
|ψ> = a|0⟩ + √(1-a²)|1⟩
"""
function density_matrix(a)
    v = [a, sqrt(1 - a^2)]
    return v * v'
end

"""
    codespace_state(n_qubits, a, b)

Create a logic qubit a|00...0> + b|11...1>,
where we adopt the convention that |00...0> is at index 1
of the 2^n_qubits state vector and |11...1> is at index n_qubits
"""
function codespace_state(n_qubits, a, b, c, d)
    psi = zeros(ComplexF64, 2^n_qubits)
    psi[1] = a
    psi[2] = c
    psi[4] = d
    psi[end] = b
    psi = normalize(psi)
    return psi * psi'
end



"""
    get_amplitudedamping_operators(gamma, t)
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
    get_dephasing_operators(gamma, t)
"""
function get_dephasing_operators(gamma, t)
    e = exp(-gamma*t)
    p = (1.0 - e) / 2.0

    k1 = sqrt(1.0 - p) * I2
    k2 = sqrt(p) * X
    
    return [k1, k2]
end

"""
    get_bitflip_operators(gamma, t)
"""
function get_bitflip_operators(gamma, t)
    e = exp(-2*gamma*t)
    p = (1.0 - e) / 2.0

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

    for qubit in n_qubits
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
    recovery_map(gamma, t, σ, ρ)
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
function get_kraus_from_map(E::Function, d::Int; tol=1e-10)
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
    for k in 1:length(evals)
        λ = evals[k]
        if λ > tol
            v = evecs[:, k]              # length d^2
            K = reshape(v, d, d)         # d × d matrix
            push!(kraus, sqrt(λ) * K)
        end
    end

    return kraus
end
