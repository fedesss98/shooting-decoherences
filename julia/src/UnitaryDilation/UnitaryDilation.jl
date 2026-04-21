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
    ancilla_ground_state(T, d_a)

Return |0><0| as a d_a × d_a density matrix of element type T.
"""
function ancilla_ground_state(::Type{T}, d_a::Int) where T
    ρa = zeros(T, d_a, d_a)
    ρa[1, 1] = one(T)
    return ρa
end

"""
    ancilla_thermal_qubit(alpha; T=Float64)

Return the qubit diagonal state diag(alpha, 1-alpha).
"""
function ancilla_thermal_qubit(alpha::Real; T::Type=Float64)
    0 <= alpha <= 1 || throw(ArgumentError("alpha must satisfy 0 ≤ alpha ≤ 1"))
    return Matrix{T}(Diagonal(T[alpha, 1 - alpha]))
end

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
