module UnitaryDilation

using LinearAlgebra

export 
CollisionModel, apply_collision, extract_kraus_operators, 
partial_traces, ptrace_ancilla, ptrace_system,
unitary_dilation

include("unitary_dilation.jl")
include("extract_kraus.jl")
include("partial_traces.jl")
include("collisionmodel.jl")


"""
    apply_collision(model, rho)

Simulates the recovery map by:
1. Initializing ancilla in |0><0|
2. Applying the collision unitary U
Returns a Tuple:
 - The recovered state after partial trace over ancilla
 - The modified ancilla state after interaction
"""
function apply_collision(model::CollisionModel, rho::Matrix{T}; trace::Bool=false) where T
    d_s = model.dim_sys
    d_a = model.dim_anc
    
    # 1. Prepare Joint State: ρ ⊗ |0><0|_anc
    # Ancilla |0> is the first basis vector [1, 0, ...]
    anc_0 = zeros(T, d_a, d_a)
    anc_0[1, 1] = 1.0
    
    rho_total = kron(rho, anc_0)
    
    # 2. Apply Unitary: U (ρ ⊗ |0><0|) U†
    rho_after = model.U * rho_total * model.U'
    
    if trace
        rho_sys_out, rho_anc_out = partial_traces(rho_after, d_s, d_a)
        return (rho_sys_out, rho_anc_out)
    end
    
    return rho_after
end


end # module
