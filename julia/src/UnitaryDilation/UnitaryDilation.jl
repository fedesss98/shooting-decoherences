module UnitaryDilation

using LinearAlgebra

export 
PetzCollisionModel, apply_petz_collision, extract_kraus_operators

include("petzcollisionmodel.jl")
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
    d, m = model.dim_sys, model.dim_anc

    
    # Prepare Joint State: ρ ⊗ |0><0|_anc
    # Ancilla |0> is the first basis vector [1, 0, ...]
    anc0 = zeros(T, m)
    anc0[1] = one(T)
    rho_anc = anc0 * anc0'
    
    joint_rho = kron(rho, rho_anc)
    
    # Apply Unitary: U (ρ ⊗ |0><0|) U†
    rho_after = model.U * joint_rho * model.U'
    
    # Partial Trace over Ancilla
    # We sum over the diagonal blocks of the ancilla
    rho_rec = sum(rho_after[(i-1)*d+1:i*d, (i-1)*d+1:i*d] for i in 1:m)

    # Partial Trace over System
    rho_anc_out = sum(rho_after[i:d:end, j:d:end] for i in 1:d, j in 1:d if i==j)
    
    return rho_rec, rho_anc_out
end


end # module
