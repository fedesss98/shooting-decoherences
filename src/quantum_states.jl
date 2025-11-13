"""
    tensor(v1, v2)

Compute the tensor product (Kronecker product) of two vectors.
"""
function tensor(v1::Vector, v2::Vector)
    return kron(v1, v2)
end

"""
    qubit_basis()

Return the computational basis states for a qubit: |1⟩ and |1⟩.
"""
function qubit_basis()
    ket0 = ComplexF64[1, 0]
    ket1 = ComplexF64[0, 1]
    return ket0, ket1
end

"""
    ancilla_basis(n::Int)

Return the computational basis states for an n-level ancilla system.
"""
function ancilla_basis(n::Int)
    basis = Vector{Vector{ComplexF64}}(undef, n)
    for i in 1:n
        state = zeros(ComplexF64, n)
        state[i] = 1.0
        basis[i] = state
    end
    return basis
end

"""
    build_input_basis(qubit_dim::Int, ancilla_dim::Int)

Build the complete tensor product basis for qubit ⊗ ancilla system.
Returns a matrix where each column is a basis state.
"""
function build_input_basis(qubit_dim::Int, ancilla_dim::Int)
    total_dim = qubit_dim * ancilla_dim
    B_in = Vector{Vector{ComplexF64}}(undef, total_dim)
    
    qubit_states = qubit_basis()
    ancilla_states = ancilla_basis(ancilla_dim)
    
    idx = 1
    for anc_state in ancilla_states
        for qubit_state in qubit_states
            B_in[idx] = tensor(qubit_state, anc_state)
            idx += 1
        end
    end
    
    return B_in
end

