
"""
    kraus_operators(pa::Float, pb::Float, a1, a2, b1, b2)

Construct the Kraus Operator for the recovery map
Λ[ρ] = Tr(ρ) σ,
were the recovered state σ is defined by
pa [a1 a2] + pb[b1 b2] * [b1 b2]
"""
function kraus_operators_recovery1(pa::Float64, pb::Float64, a1::Float64, a2::Float64, b1::Float64, b2::Float64)
    K1 = sqrt(pa) * [a1 0; a2 0]
    K2 = sqrt(pb) * [b1 0; b2 0]
    K3 = sqrt(pa) * [0 a1; 0 a2]
    K4 = sqrt(pb) * [0 b1; 0 b2]

    return [K1, K2, K3, K4]
end

"""
    isometry(system_state::Vector)

Implement the action of the isomery V, acting on a state of the system
and extending the action of the map in the system+ancilla Hilbert Space.
"""
function isometry(system_state::Vector, ancilla_basis, kraus_operators)
    expanded_dims = size(system_state)[1] * size(ancilla_basis[1])[1]
    result = Vector{ComplexF64}(undef, expanded_dims)
    for (k, a) in zip(kraus_operators, ancilla_basis)
        result += kron(k * system_state, a)
    end
    return result
end
