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
    get_kraus_operators(gamma, t)
"""
function get_kraus_operators(gamma, t)
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
    amplitude_damping_channel(gamma, t, ρ)
"""
function amplitude_damping_channel(kraus, ρ)
    return sum([k*ρ*k' for k in kraus])
end


"""
    recovery_map(gamma, t, σ, ρ)
"""
function recovery_map(kraus, σ, ρ)
    # Define the evolution map and its adjoint
    map(ρ) = sum([k*ρ*k' for k in kraus])
    map_adj(ρ) = sum([k'*ρ*k for k in kraus])

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
