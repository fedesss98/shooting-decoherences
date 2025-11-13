"""
    construct_unitary(B_out::Matrix, B_in::Matrix)

Construct a unitary matrix from input and output bases.

# Arguments
- `B_out`: Output orthonormal basis (columns)
- `B_in`: Input orthonormal basis (columns)

# Returns
- Unitary matrix U such that U * B_in[:, i] = B_out[:, i]
"""
function construct_unitary(B_out::Matrix{ComplexF64}, B_in::Matrix{ComplexF64})
    return B_out * B_in'
end

"""
    verify_unitary(U::Matrix; atol=1e-10)

Verify that a matrix is unitary (U† U = I).
"""
function verify_unitary(U::Matrix{ComplexF64}; atol=1e-10)
    n = size(U, 1)
    return isapprox(U' * U, I(n), atol=atol)
end

"""
    print_state(state::Vector, label::String="State")

Pretty print a quantum state vector.
"""
function print_state(state::Vector{ComplexF64}, label::String="State")
    println("$label:")
    for (i, amp) in enumerate(state)
        if abs(amp) > 1e-10
            println("  |$(i-1)⟩: $(round(amp, digits=4))")
        end
    end
end


"""
    chop_parts(z::ComplexF64)

Set small values of a complex number to zero.
"""
function chop_parts(z::ComplexF64, tol::Float64)
    # Check real part
    r = real(z)
    new_r = abs(r) > tol ? r : 0.0
    
    # Check imaginary part
    i = imag(z)
    new_i = abs(i) > tol ? i : 0.0
    
    return complex(new_r, new_i)
end

"""
    chop!(input)

Set small elements to zero.
"""
function chop!(input, tol::Float64 = 1e-9)
    map!(z -> chop_parts(z, tol), input, input)
end

"""
    partial_trace_ancilla(rho::Matrix, dim_system::Int, dim_ancilla::Int)

Compute the partial trace over the ancilla (second subsystem).

# Arguments
- `rho`: Density matrix of the composite system (dim_system × dim_ancilla square)
- `dim_system`: Dimension of the system to keep (qubit = 2)
- `dim_ancilla`: Dimension of the ancilla to trace out (4-level = 4)

# Returns
- Reduced density matrix of the system (dim_system × dim_system)
"""
function partial_trace_ancilla(rho::Matrix{ComplexF64}, dim_system::Int, dim_ancilla::Int)
    total_dim = dim_system * dim_ancilla
    
    if size(rho) != (total_dim, total_dim)
        error("Density matrix size doesn't match system dimensions!")
    end
    
    rho_reduced = zeros(ComplexF64, dim_system, dim_system)
    
    # Sum over ancilla basis states
    for k in 0:(dim_ancilla-1)
        for i in 0:(dim_system-1)
            for j in 0:(dim_system-1)
                # Map indices: |i⟩_S ⊗ |k⟩_A has index i*dim_ancilla + k + 1 (Julia 1-indexed)
                row_idx = i * dim_ancilla + k + 1
                col_idx = j * dim_ancilla + k + 1
                
                rho_reduced[i+1, j+1] += rho[row_idx, col_idx]
            end
        end
    end
    
    return rho_reduced
end

