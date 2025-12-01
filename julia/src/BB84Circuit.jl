module BB84Circuit

using LinearAlgebra

export encode_bit, decode_bit, fidelity

include("utils.jl")
include("metrics.jl")

"""
    encode_bit(value, basis)
# Arguments
- `value`: value of the bit to encode (0 or 1)
- `basis`: Pauli basis to encode the bit (X or Y)
# Returns
- `psi`: quantum state encoding the bit value in its eigenvalue given the correct `basis`
"""
function encode_bit(value, basis)
    psi_0 = ComplexF64[1, 0]
    psi_1 = ComplexF64[0, 1]
    if basis == "Z"
        if value == 1
            psi = psi_1
        else
            psi = psi_0
        end
    else
        if value == 0
            psi = (psi_0 + psi_1) / sqrt(2)
        else
            psi = (psi_0 - psi_1) / sqrt(2)
        end
    end

    return psi
end

"""
    decode_bit(state, basis)
# Arguments
- `state`: incoming state (ket vector or 2x2 density matrix) to be measured to get the decoded bit (0 or 1)
- `basis`: Pauli basis to measure the state (X or Y)
# Returns
- `bit`: the bit value
"""
function decode_bit(state::Matrix{ComplexF64}, basis::Union{Symbol, String})
    basis = Symbol(basis)  # Use Symbol internally
    if basis == :Z
        prob_0 = real(state[1, 1])
        prob_1 = real(state[2, 2])
        outcomes = [0, 1]
    elseif basis == :X
        # Formula for <+|rho|+>
        prob_0 = 0.5 * real(state[1,1] + state[2,2] + state[1,2] + state[2,1])
        prob_1 = 1.0 - prob_0
        outcomes = ["+", "-"]
    else
        error("Basis must be :Z or :X")
    end

    # Give the measurement result as a stochastic process
    r = rand()
    result = r < prob_0 ? outcomes[1] : outcomes[2]
    
    return result
end

function decode_bit(state::Vector{ComplexF64}, basis::Union{Symbol, String})
    basis = Symbol(basis)  # Use Symbol internally
    normalize!(state)  # Ensure state is normalized
    if basis == :Z
        # Probability of outcome 0 (state |0>) is |ψ[1]|²
        prob_0 = abs2(state[1])
    elseif basis == :X
        # The basis states are |+> (outcome 0) and |-> (outcome 1)
        # Probability of |+> is |⟨+|ψ⟩|²
        # ⟨+| = [1 1] / √2
        inner_prod = (state[1] + state[2]) / sqrt(2)
        prob_0 = abs2(inner_prod)
        
    else
        error("Basis must be :Z or :X")
    end

    result = rand() < prob_0 ? 0 : 1
    return result
end
end
