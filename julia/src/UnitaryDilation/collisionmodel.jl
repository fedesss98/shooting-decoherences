
"""
    CollisionModel

A struct to organize the simulation of a recovery map via a collision model.
It stores the system/ancilla dimensions, the reference state, the forward/recovery Kraus ops,
and the resulting Stinespring dilation Unitary.
"""
struct CollisionModel{T<:Number}
    dim_sys::Int
    dim_anc::Int
    sigma::Matrix{T}              # Reference state
    kraus_fwd::Vector{Matrix{T}}  # Forward noise {K_i}
    kraus_rec::Vector{Matrix{T}}  # Petz recovery {R_i}
    U::Matrix{T}                  # The collision unitary
end

"""
    CollisionModel(kraus_fwd, sigma)

General constructor. Given arbitrary Kraus operators and a reference state:
1. Computes the Petz Recovery Kraus operators.
2. Constructs the Stinespring Collision Unitary U.
"""
function CollisionModel(kraus_fwd::Vector{Matrix{T1}}, sigma::Matrix{T2}; n::Int=1) where {T1<:Number, T2<:Number}
    # Determine a common type
    T = promote_type(T1, T2)
    # and convert inputs to this common type
    sigma = Matrix{T}(sigma)
    kraus_fwd = Matrix{T}[Matrix{T}(K) for K in kraus_fwd]
    # Optionally expand single-qubit Kraus to n-qubits if needed
    if n > 1
        expand_kraus_operators!(kraus_fwd, n)
    end
    d_sys = size(sigma, 1)
    d_anc = length(kraus_fwd) # Ancilla dimension = number of Kraus ops
    
    # 1. Compute Forward Action on Sigma: N(σ) = ∑ K σ K'
    N_sigma = sum(K * sigma * K' for K in kraus_fwd)

    # 2. Compute Petz Recovery Kraus Ops: R_i = σ^(1/2) K_i' N(σ)^(-1/2)
    # Note: Using pinv for stability, assuming Hermitian matrices
    σ_sqrt = sqrt(Hermitian(sigma))
    σ_out_inv_sqrt = inv(sqrt(Hermitian(N_sigma + 1e-10*I)))
    kraus_rec = [σ_sqrt * K' * σ_out_inv_sqrt for K in kraus_fwd]

    # 3. Generate the Collision Unitary U
    # Total dimension = System ⊗ Ancilla
    d_tot = d_sys * d_anc
    
    # We construct the first d_sys columns of U.
    # These columns define the action U(ρ ⊗ |0><0|) U'.
    # Column j corresponds to input state |j>_S ⊗ |0>_E.
    # Formula: U |j, 0> = ∑_i (R_i |j>) ⊗ |i>_E
    
    # Pre-allocate the "known" part of the unitary (V)
    V = zeros(T, d_tot, d_sys)
    
    # System basis vectors
    sys_basis = [zeros(T, d_sys) for _ in 1:d_sys]
    for k in 1:d_sys; sys_basis[k][k] = 1.0; end

    # Ancilla basis vectors
    anc_basis = [zeros(T, d_anc) for _ in 1:d_anc]
    for k in 1:d_anc; anc_basis[k][k] = 1.0; end

    for j in 1:d_sys
        input_vec = sys_basis[j] # |j>
        
        # Calculate the resulting vector in the joint space
        output_vec_joint = zeros(T, d_tot)
        
        for i in 1:d_anc
            # Apply Recovery operator i to system state j
            transformed_sys = kraus_rec[i] * input_vec
            
            # Tensor product with ancilla state |i>
            # joint index = (sys_idx - 1) * d_anc + anc_idx (Julia uses Column Major, but kron is standard)
            # We use Julia's kron: kron(A, B) computes A ⊗ B.
            # Here we need transformed_sys ⊗ |i>_anc
            term = kron(transformed_sys, anc_basis[i])
            output_vec_joint += term
        end
        
        V[:, j] = output_vec_joint
    end

    # 4. Complete the Unitary
    # We have V (d_tot x d_sys) which is isometric (V'V = I).
    # We need to fill the remaining columns to make it square and unitary.
    # QR decomposition is a numerically stable way to do this.
    # If A = QR, and A has orthonormal columns, Q's first columns are A.
    Q_fact = qr(V)
    # FORCE FULL SQUARE MATRIX:
    # Multiply Q (which acts like a operator) by the full Identity matrix
    U_full = Q_fact.Q * Matrix{T}(I, d_tot, d_tot)

    return CollisionModel(d_sys, d_anc, sigma, kraus_fwd, kraus_rec, U_full)
end


"""
    CollisionModel(noise_channel::Function, gamma, sigma; nsteps=100)

Constructor for when Kraus operators are unknown. The noise channel is given as
a function `noise_channel(rho, gamma)` that returns the output state.

The Petz recovery is constructed by numerical Choi-Jamiolkowski:
- Discretize gamma ∈ [0, gamma] into `nsteps`
- Compute dΦ/dγ ≈ (Φ(γ+dγ) - Φ(γ))/dγ
- Extract Kraus operators via eigendecomposition of Choi matrix
"""
function CollisionModel(noise_channel::Function, gamma::Real, sigma::Matrix{T}; 
                            nsteps::Int=100) where T<:Number
    d = size(sigma, 1)
    
    # Maximize basis for Choi matrix
    max_ent = sum(kron(Matrix{T}(I, d, d)[:, i:i], Matrix{T}(I, d, d)[:, i:i]) 
                  for i in 1:d) / sqrt(d)
    rho_max = max_ent * max_ent'
    
    # Apply channel to maximally entangled state
    # Φ ⊗ I acting on |Ψ⟩⟨Ψ| gives Choi matrix
    function choi_matrix(γ)
        choi = zeros(T, d*d, d*d)
        for i in 1:d, j in 1:d
            input = zeros(T, d, d)
            input[i, j] = one(T)
            output = noise_channel(input, γ)
            for k in 1:d, l in 1:d
                choi[(i-1)*d+k, (j-1)*d+l] = output[k, l]
            end
        end
        return choi
    end
    
    # Derivative at gamma
    dγ = gamma / nsteps
    J = (choi_matrix(gamma + dγ) - choi_matrix(gamma)) / dγ
    J = (J + J') / 2  # Symmetrize
    
    # Extract Kraus from Choi eigendecomposition
    vals, vecs = eigen(Hermitian(J))
    kraus_fwd = Matrix{T}[]
    for i in 1:d*d
        if real(vals[i]) > 1e-12
            K = sqrt(abs(vals[i])) * reshape(vecs[:, i], d, d)
            push!(kraus_fwd, K)
        end
    end
    
    isempty(kraus_fwd) && push!(kraus_fwd, zeros(T, d, d))
    
    CollisionModel(kraus_fwd, sigma)
end

"""
    CollisionModel(M::Matrix, sigma::Matrix)

Constructor from superoperator M (d²×d² matrix in column-stacking convention).
Extracts Kraus operators via Choi matrix eigendecomposition.
"""
function CollisionModel(M::Matrix{T}, sigma::Matrix{T}) where T<:Number
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
    
    CollisionModel(kraus_fwd, sigma)
end

"""
    CollisionModel(kraus_single::Vector{Matrix}, sigma::Matrix, N::Int)

Constructor for N-qubit system where single-qubit Kraus operators act independently.
`kraus_single` are 2×2 Kraus operators acting on each qubit.
The total system has dimension 2^N × 2^N.

The full Kraus operators are tensor products over all qubits.
"""
function CollisionModel(kraus_single::Vector{Matrix{T}}, sigma::Matrix{T}, N::Int) where T<:Number
    d_single = size(kraus_single[1], 1)
    @assert d_single == 2 "Single-qubit Kraus operators must be 2×2"
    @assert size(sigma, 1) == 2^N "Sigma dimension must match 2^N"
    
    m = length(kraus_single)
    
    # Generate all tensor product combinations
    # For N qubits with m Kraus ops each: m^N total Kraus operators
    kraus_fwd = Matrix{T}[]
    
    function tensor_product_kraus(indices)
        K = kraus_single[indices[1]]
        for i in 2:N
            K = kron(K, kraus_single[indices[i]])
        end
        return K
    end
    
    # All combinations of indices (Cartesian product)
    for idx in Iterators.product([1:m for _ in 1:N]...)
        push!(kraus_fwd, tensor_product_kraus(collect(idx)))
    end
    
    CollisionModel(kraus_fwd, sigma)
end


"""
    expand_kraus_operators!(kraus_single::Vector{Matrix{T}}, N::Int) where T

Expand single-qubit Kraus operators to N-qubit system via tensor products.
Modifies the input vector in place, replacing it with all m^N tensor product combinations.
"""
function expand_kraus_operators!(kraus_single::Vector{Matrix{T}}, N::Int) where T    
    d_single = size(kraus_single[1], 1)
    @assert d_single == 2 "Single-qubit Kraus operators must be 2×2"
    
    m = length(kraus_single)
    kraus_expanded = Matrix{T}[]
    
    # Generate all tensor product combinations
    for idx in Iterators.product([1:m for _ in 1:N]...)
        K = kraus_single[idx[1]]
        for i in 2:N
            K = kron(K, kraus_single[idx[i]])
        end
        push!(kraus_expanded, K)
    end
    
    # Modify in place
    empty!(kraus_single)
    append!(kraus_single, kraus_expanded)
    
    return kraus_single
end