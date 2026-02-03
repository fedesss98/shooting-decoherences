module UnitaryDilation

using LinearAlgebra

export 
PetzCollisionModel, apply_petz_collision

include("extract_kraus.jl")
include("extract_kraus.jl")

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
