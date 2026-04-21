module UnitaryDilation

using LinearAlgebra

export 
CollisionModel, apply_collision, extract_kraus_operators, 
partial_traces, ptrace_ancilla, ptrace_system,
unitary_dilation,
ancilla_ground_state, ancilla_thermal_qubit

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
    apply_collision(model, rho; trace=false, ancilla_state=nothing, alpha=nothing)

Apply the joint collision unitary to `rho ⊗ ancilla_state`.

Defaults:
- if both `ancilla_state` and `alpha` are omitted, uses |0><0| on the ancilla.
- if `alpha` is provided, uses the qubit state diag(alpha, 1-alpha).

Returns:
- full joint output state if `trace=false`
- `(rho_sys_out, rho_anc_out)` if `trace=true`
"""
function apply_collision(
    model::CollisionModel,
    rho::AbstractMatrix{T};
    trace::Bool=false,
    ancilla_state::Union{Nothing,AbstractMatrix}=nothing,
    alpha::Union{Nothing,Real}=nothing,
) where T
    d_s = model.dim_sys
    d_a = model.dim_anc

    size(rho) == (d_s, d_s) || throw(ArgumentError("rho must be of size $d_s × $d_s"))

    ρa =
        if ancilla_state !== nothing && alpha !== nothing
            throw(ArgumentError("provide either ancilla_state or alpha, not both"))
        elseif ancilla_state !== nothing
            Matrix{T}(ancilla_state)
        elseif alpha !== nothing
            d_a == 2 || throw(ArgumentError("alpha is only supported for qubit ancillae (dim_anc = 2)"))
            Matrix{T}(ancilla_thermal_qubit(alpha; T=T))
        else
            ancilla_ground_state(T, d_a)
        end

    size(ρa) == (d_a, d_a) || throw(ArgumentError("ancilla state must be of size $d_a × $d_a"))

    rho_total = kron(Matrix{T}(rho), ρa)
    rho_after = model.U * rho_total * model.U'

    if trace
        rho_sys_out, rho_anc_out = partial_traces(rho_after, d_s, d_a)
        return rho_sys_out, rho_anc_out
    end

    return rho_after
end


end # module
