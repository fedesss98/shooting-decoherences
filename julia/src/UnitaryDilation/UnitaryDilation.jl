module UnitaryDilation

using LinearAlgebra

export 
PetzCollisionModel, apply_petz_collision


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

    d_sys = size(sigma, 1)
    d_anc = length(kraus_fwd) # Ancilla dimension = number of Kraus ops
    
    # 1. Compute Forward Action on Sigma: N(σ) = ∑ K σ K'
    N_sigma = zeros(T, d_sys, d_sys)
    for K in kraus_fwd
        N_sigma += K * sigma * K'
    end

    # 2. Compute Petz Recovery Kraus Ops: R_i = σ^(1/2) K_i' N(σ)^(-1/2)
    # Note: Using pinv for stability, assuming Hermitian matrices
    sqrt_sigma = sqrt(sigma)
    inv_sqrt_N_sigma = pinv(sqrt(N_sigma)) 
    
    kraus_rec = Matrix{T}[]
    for K in kraus_fwd
        # The Petz formula
        R = sqrt_sigma * K' * inv_sqrt_N_sigma
        push!(kraus_rec, R)
    end

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

    return PetzCollisionModel(d_sys, d_anc, sigma, kraus_fwd, kraus_rec, U_full)
end


"""
    apply_petz_collision(model, rho)

Simulates the recovery map by:
1. Initializing ancilla in |0><0|
2. Applying the collision unitary U
Returns a Tuple:
 - The recovered state after partial trace over ancilla
 - The modified ancilla state after interaction
"""
function apply_petz_collision(model::PetzCollisionModel, rho::Matrix{T}) where T
    d_s = model.dim_sys
    d_a = model.dim_anc
    
    # 1. Prepare Joint State: ρ ⊗ |0><0|_anc
    # Ancilla |0> is the first basis vector [1, 0, ...]
    anc_0 = zeros(T, d_a, d_a)
    anc_0[1, 1] = 1.0
    
    joint_rho = kron(rho, anc_0)
    
    # 2. Apply Unitary: U (ρ ⊗ |0><0|) U†
    rho_prime_joint = model.U * joint_rho * model.U'
    
    # 3. Partial Trace over Ancilla
    # We sum over the diagonal blocks of the ancilla
    rho_out = zeros(T, d_s, d_s)
    
    # Indexing logic for Partial Trace (System ⊗ Ancilla)
    # Block (i, j) of size d_a x d_a corresponds to system indices i, j
    # But standard kron(S, A) interleaves differently. 
    # Let's do it explicitly based on indices:
    # Index I in joint space = (i_sys - 1)*d_anc + i_anc
    rho_sys_out = zeros(T, d_s, d_s)
    
    for i in 1:d_s
        for j in 1:d_s
            # Locate the (i, j) block of size d_a x d_a
            row_start = (i - 1) * d_a + 1
            col_start = (j - 1) * d_a + 1
            
            # Trace this block
            val = zero(T)
            for k in 0:(d_a - 1)
                val += rho_prime_joint[row_start + k, col_start + k]
            end
            rho_sys_out[i, j] = val
        end
    end

    # Partial Trace over System
    # (Sum the diagonal blocks)
    sigma_anc_out = zeros(T, d_a, d_a)
    
    for i in 1:d_s
        # Extract the i-th diagonal block corresponding to system index i
        idx_start = (i - 1) * d_a + 1
        idx_end   = i * d_a
        
        sigma_anc_out += rho_prime_joint[idx_start:idx_end, idx_start:idx_end]
    end
    
    return (rho_sys_out, sigma_anc_out)
end


end # module
