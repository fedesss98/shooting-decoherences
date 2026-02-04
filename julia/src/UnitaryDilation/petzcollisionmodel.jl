
"""
    PetzCollisionModel

A struct to organize the simulation of a Petz recovery map via a collision model.
It stores the system/ancilla dimensions, the reference state, the forward/recovery Kraus ops,
and the resulting Stinespring dilation Unitary.
"""
struct PetzCollisionModel{T<:Number}
    dim_sys::Int
    dim_anc::Int
    sigma::Matrix{T}              # Reference state
    kraus_fwd::Vector{Matrix{T}}  # Forward noise {K_i}
    kraus_rec::Vector{Matrix{T}}  # Petz recovery {R_i}
    U::Matrix{T}                  # The collision unitary
end

"""
    PetzCollisionModel(kraus_fwd, sigma)

General constructor. Given arbitrary Kraus operators and a reference state:
1. Computes the Petz Recovery Kraus operators.
2. Constructs the Stinespring Collision Unitary U.
"""
function PetzCollisionModel(kraus_fwd::Vector{Matrix{T1}}, sigma::Matrix{T2}) where {T1<:Number, T2<:Number}
    # Determine a common type
    T = promote_type(T1, T2)
    # and convert inputs to this common type
    sigma = Matrix{T}(sigma)
    kraus_fwd = [Matrix{T}(K) for K in kraus_fwd]

    d = size(kraus_fwd[1], 1)
    m = length(kraus_fwd)
    
    # Compute Forward Action on Sigma: N(σ) = ∑ K σ K'
    N_sigma = sum(K * sigma * K' for K in kraus_fwd)
    
    # 2. Compute Petz Recovery Kraus Ops: R_i = σ^(1/2) K_i' N(σ)^(-1/2)
    # Note: Using pinv for stability, assuming Hermitian matrices
    sqrt_sigma = sqrt(Hermitian(sigma))
    inv_sqrt_N_sigma = inv(sqrt(Hermitian(N_sigma + 1e-10*I)))
    
    kraus_rec = [sqrt_sigma * K' * inv_sqrt_N_sigma for K in kraus_fwd]

    # 3. Generate the Collision Unitary U
    # Total dimension = System ⊗ Ancilla
    d_tot = d * m
    
    # We construct the first d columns of U.
    # These columns define the action U(ρ ⊗ |0><0|) U'.
    # Column j corresponds to input state |j>_S ⊗ |0>_E.
    # Formula: U |j, 0> = ∑_i (R_i |j>) ⊗ |i>_E
    
    # Stinespring unitary
    U = zeros(T, d*m, d*m)
    for i in 1:m
        U[1:d, (i-1)*d+1:i*d] = kraus_rec[i]
    end
    
    # Complete to unitary
    Q = Matrix(qr(U).Q)
    U = size(Q) == (d*m, d*m) ? Q : [U zeros(T, d*m, d*m - size(U,2))]
    
    PetzCollisionModel(d, m, sigma, kraus_fwd, kraus_rec, U)
end


"""
    PetzCollisionModel(noise_channel::Function, gamma, sigma)

Constructor for when Kraus operators are unknown. The noise channel is given as
a function `noise_channel(rho, gamma)` that applies the channel with parameter gamma.

Extracts Kraus operators via Choi-Jamiolkowski isomorphism:
- Construct Choi matrix by applying channel to each basis element
- Eigendecompose to get Kraus operators
"""
function PetzCollisionModel(noise_channel::Function, sigma::Matrix{T}) where T<:Number
    d = size(sigma, 1)
    
    # Construct Choi matrix: J = sum_ij |i⟩⟨j| ⊗ Φ(|i⟩⟨j|)
    choi = zeros(T, d*d, d*d)
    for i in 1:d, j in 1:d
        input = zeros(T, d, d)
        input[i, j] = one(T)
        output = noise_channel(input)
        for k in 1:d, l in 1:d
            choi[(i-1)*d+k, (j-1)*d+l] = output[k, l]
        end
    end
    
    # Symmetrize for numerical stability
    choi = (choi + choi') / 2
    
    # Extract Kraus from Choi eigendecomposition
    vals, vecs = eigen(Hermitian(choi))
    kraus_fwd = Matrix{T}[]
    for i in 1:d*d
        if real(vals[i]) > 1e-12
            K = sqrt(abs(vals[i])) * reshape(vecs[:, i], d, d)
            push!(kraus_fwd, K)
        end
    end
    
    isempty(kraus_fwd) && push!(kraus_fwd, zeros(T, d, d))
    
    PetzCollisionModel(kraus_fwd, sigma)
end


"""
    PetzCollisionModel(M::Matrix, sigma::Matrix)

Constructor from superoperator M (d²×d² matrix in column-stacking convention).
Extracts Kraus operators via Choi matrix eigendecomposition.
"""
function PetzCollisionModel(M::Matrix{T}, sigma::Matrix{T}) where T<:Number
    d = size(sigma, 1)
    
    # Choi matrix from superoperator: reshape and transpose
    # J = sum_ij |i⟩⟨j| ⊗ M(|i⟩⟨j|)
    choi = reshape(M, d, d, d, d)
    choi = permutedims(choi, [1, 3, 2, 4])
    choi = reshape(choi, d*d, d*d)
    
    # Symmetrize
    choi = (choi + choi') / 2
    
    # Extract Kraus operators
    vals, vecs = eigen(Hermitian(choi))
    kraus_fwd = Matrix{T}[]
    for i in 1:d*d
        if real(vals[i]) > 1e-12
            K = sqrt(abs(vals[i])) * reshape(vecs[:, i], d, d)
            push!(kraus_fwd, K)
        end
    end
    
    isempty(kraus_fwd) && push!(kraus_fwd, zeros(T, d, d))
    
    PetzCollisionModel(kraus_fwd, sigma)
end