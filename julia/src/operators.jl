const I2 = ComplexF64[1.0+0.0im 0.0; 0.0 1.0]
const Z  = ComplexF64[1.0+0.0im 0.0; 0.0 -1.0]
const X  = ComplexF64[0.0im 1.0; 1.0 0.0]

"""
    tensor_power(A, n)
Returns the n-fold tensor product of A with itself, i.e. A ⊗ A ⊗ ... ⊗ A (n times).
"""
function tensor_power(A, n::Int)
    @assert n ≥ 1
    return foldl(kron, [A for _ in 1:n])
end


"""
    get_depolarizing_operators(gamma, t; n_qubits=1)

Returns Kraus operators for a global/correlated depolarizing-like Pauli channel.

For n_qubits = 1, this is the usual single-qubit depolarizing channel.

For n_qubits > 1, this describes fully correlated Pauli noise:
    with probability 1-p: no error
    with probability p/3: X⊗X⊗...⊗X
    with probability p/3: Y⊗Y⊗...⊗Y
    with probability p/3: Z⊗Z⊗...⊗Z
"""
function get_depolarizing_operators(gamma, t; n_qubits=1)
    p = 1.0 - exp(-gamma * t)

    K0 = sqrt(1.0 - p) * tensor_power(Id2, n_qubits)
    K1 = sqrt(p / 3.0) * tensor_power(X, n_qubits)
    K2 = sqrt(p / 3.0) * tensor_power(Y, n_qubits)
    K3 = sqrt(p / 3.0) * tensor_power(Z, n_qubits)

    return [K0, K1, K2, K3]
end


"""
    get_amplitudedamping_operators(gamma, t; n_qubits=1)

Returns Kraus operators for a global/correlated amplitude damping channel.

For n_qubits = 1, this is the usual amplitude damping channel.

For n_qubits > 1, this implements collective decay:
    |11...1⟩ → |00...0⟩

This is one possible correlated amplitude damping model, not the only one.
"""
function get_amplitudedamping_operators(gamma, t; n_qubits=1)
    e = exp(-gamma * t)

    dim = 2^n_qubits

    K0 = Matrix{ComplexF64}(I, dim, dim)
    K0[dim, dim] = sqrt(e)

    K1 = zeros(ComplexF64, dim, dim)
    K1[1, dim] = sqrt(1.0 - e)

    return [K0, K1]
end

"""
    get_phasedamping_operators(gamma, t; n_qubits=1)

Returns Kraus operators for a global/correlated phase damping channel.

For n_qubits = 1, this is the usual phase damping channel.

For n_qubits > 1, this damps coherences involving |11...1⟩ collectively.
This is one possible correlated phase damping model, not the only one.
"""
function get_phasedamping_operators(gamma, t; n_qubits=1)
    e = exp(-gamma * t)

    dim = 2^n_qubits

    K0 = Matrix{ComplexF64}(I, dim, dim)
    K0[dim, dim] = sqrt(e)

    K1 = zeros(ComplexF64, dim, dim)
    K1[dim, dim] = sqrt(1.0 - e)

    return [K0, K1]
end

"""
    get_bitflip_operators(gamma, t)
See for example
    https://docs.pennylane.ai/en/stable/code/api/pennylane.BitFlip.html
"""
function get_bitflip_operators(gamma, t; n_qubits=1)
    p = (1.0 - exp(-gamma*t)) / 2.0

    K_no_flip = sqrt(1.0 - p) * foldl(kron, [I2 for _ in 1:n_qubits])
    K_all_flip = sqrt(p) * foldl(kron, [X for _ in 1:n_qubits])

    return [K_no_flip, K_all_flip]
end

"""
    apply_channel(kraus, ρ)
"""
function apply_channel(kraus, ρ)
    return sum([k*ρ*k' for k in kraus])
end

"""
    expand_kraus_operators(kraus, n_qubits)

Return Kraus operators acting on the full n-qubit system. Single-qubit
operators are expanded as independent tensor-product noises on every qubit.
Operators already acting on the full system are returned unchanged.
"""
function expand_kraus_operators(kraus, n_qubits::Int)
    target_dim = 2^n_qubits
    op_dim = size(kraus[1], 1)

    all(size(K) == (op_dim, op_dim) for K in kraus) ||
        throw(ArgumentError("All Kraus operators must have the same square dimension"))

    if op_dim == target_dim
        return [Matrix{ComplexF64}(K) for K in kraus]
    elseif op_dim != 2
        throw(ArgumentError("Kraus operators have dimension $op_dim, expected 2 or $target_dim"))
    end

    expanded = Matrix{ComplexF64}[]
    for idx in Iterators.product([eachindex(kraus) for _ in 1:n_qubits]...)
        K = Matrix{ComplexF64}(kraus[idx[1]])
        for i in 2:n_qubits
            K = kron(K, kraus[idx[i]])
        end
        push!(expanded, K)
    end

    return expanded
end

"""
    apply_channel(kraus, ρ, n_qubits; extra_dims=0)
Applies the amplitude damping channel for all n qubits in the system sequentially.
Optionally acts as the identity in the last subspace with dimentions `extra_dims`.
"""
function apply_channel(kraus, ρ, n_subsystems::Int; susbsystem_dim::Int=2, extra_dims::Int=0)
    ρi = copy(ρ)
    
    Isubsys = Matrix{ComplexF64}(I, susbsystem_dim, susbsystem_dim)
    Iextradims = Matrix{ComplexF64}(I, extra_dims, extra_dims)

    for subsystem in 1:n_subsystems
        ρf = zero(ρi)
        for k in kraus
            # Extend the Kraus operator
            op_list = [i == subsystem ? k : Isubsys for i in 1:n_subsystems]
            if extra_dims > 0
              push!(op_list, Iextradims)
            end
            k_full = foldl(kron, op_list)

            ρf += k_full * ρi * k_full'
        end
        # Update state
        ρi = ρf
    end

    return ρi
end

"""
    apply_extended_channel(ρ, kraus, origin_dim)
"""
function apply_extended_channel(ρ, kraus, origin_dim)
    dim_to_extend = size(ρ, 1) ÷ origin_dim
    kraus_extended = [kron(k, I(dim_to_extend)) for k in kraus]
    return sum([k*ρ*k' for k in kraus_extended])
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
