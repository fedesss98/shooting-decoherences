"""
Create the collisional Unitary Evolution that gives back a recovery map on the system.
"""

using DecoKiller.UnitaryDilation
using LinearAlgebra

# System Variables
d_S = 2  # System dimension (qubit)
d_A = 4  # Ancilla dimension

a1 = 1.0
a2 = 0.0
b1 = 1.0
b2 = 0.0

pa = 1/2
pb = 1/2

target_state = pa * [a1, a2] * [a1, a2]' + pb * [b1, b2] * [b1, b2]' 

kraus_operators = kraus_operators_recovery1(pa, pb, a1, a2, b1, b2)
qubit_states = qubit_basis()
ancilla_states = ancilla_basis(d_A)

B_in = build_input_basis(d_S, d_A)

function guess_output_vectors(qubit_states, ancilla_states, kraus_operators)
    """Create the set of output orthonormal vectors"""
    # Start creating the known outputs
    known_in = qubit_states
    B_out = Vector{Vector{ComplexF64}}(undef, d_S * d_A)
    idx = 1
    for inpt in known_in
        B_out[idx] = isometry(inpt, ancilla_states, kraus_operators)
        idx += 1
    end
    # Append the next candidates to be the unknown outputs of other inputs
    for sk in qubit_states
        for a in ancilla_states[2:end]
            B_out[idx] = tensor(sk, a)
            idx += 1
        end
    end


    return [chop!(v) for v in B_out] 
end

"""
    find_orthonormal_outputs(candidates::Vector{Vector{ComplexF64}})


"""
function find_orthonormal_outputs!(known, candidates)
    for (i, v) in enumerate(candidates)
        v_ort = copy(v)
        for u in known
            v_ort -= dot(u, v) * u
        end

        if norm(v_ort) == 0.0
            throw(ErrorException("Linear dependent vector among guesses!"))
        end

        v_ortnorm = v_ort / norm(v_ort)
        push!(known, v_ortnorm)
    end 

    return known
end

function build_unitary_matrix(inputs, outputs)
    d_SA = length(inputs)
    U = Matrix{ComplexF64}(undef, d_SA, d_SA)
    idx = 1
    for (input, output) in zip(inputs, outputs)
        U += output * input' 
    end

    return U
end

B_cand = guess_output_vectors(qubit_states, ancilla_states, kraus_operators)
B_out = find_orthonormal_outputs(B_cand[1:d_S], B_cand[d_S + 1:end])
U = build_unitary_matrix(B_in, B_out)

display(U)
# Unitary Check
dimension = tr(U' * U)
println("Trace U'U = $dimension")

# Create a random qubit state
psi = randn(ComplexF64, 2)
psi = psi / norm(psi)
rho_collision = U * kron(psi * psi', ancilla_states[1] * ancilla_states[1]')
recovered = partial_trace_ancilla(rho, d_S, d_A) 

