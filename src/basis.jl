"""
    complete_basis_qr(known_vectors::Matrix, total_dim::Int)

Complete an orthonormal basis using QR decomposition.

# Arguments
- `known_vectors`: Matrix where each column is a known orthonormal vector
- `total_dim`: Total dimension of the Hilbert space

# Returns
- Complete orthonormal basis as a matrix (columns are basis vectors)
"""
function complete_basis_qr(known_vectors::Matrix{ComplexF64}, total_dim::Int)
    d = size(known_vectors, 2)
    
    if d >= total_dim
        @warn "Already have a complete basis!"
        return known_vectors
    end
    
    # Create random vectors to complete the basis
    random_vecs = randn(ComplexF64, total_dim, total_dim - d)
    combined = hcat(known_vectors, random_vecs)
    
    # QR decomposition
    Q, R = qr(combined)
    
    return Matrix(Q)
end

"""
    verify_orthonormality(basis::Matrix; atol=1e-10)

Verify that a basis is orthonormal.

Returns true if basis is orthonormal within tolerance.
"""
function verify_orthonormality(basis::Matrix{ComplexF64}; atol=1e-10)
    gram = basis' * basis
    n = size(basis, 2)
    return isapprox(gram, I(n), atol=atol)
end
