"""
    extract_kraus_operators(channel_func::Function, dim::Int)

Numerically extracts the Kraus operators of a black-box channel using the Choi isomorphism.
Args:
    channel_func: A function f(rho) -> rho_out
    dim: The dimension of the system (e.g., 2 for a qubit)
"""
function extract_kraus_operators(channel_func::Function, dim::Int; threshold=1e-10)
    # 1. Create the unnormalized Maximally Entangled State |Φ+><Φ+|
    # The space is System ⊗ Reference (both size dim)
    # |Φ+> = sum(|i> ⊗ |i>)
    
    d2 = dim * dim
    phi_plus = zeros(ComplexF64, d2, d2)
    
    # Construct |Φ+><Φ+| explicitly
    # Basis index for |i>|j> is (i-1)*dim + j + 1
    for i in 1:dim
        for j in 1:dim
            # We want the term |i>|i> <j|<j|
            # In vector indices (1-based):
            row_idx = (i - 1) * dim + i
            col_idx = (j - 1) * dim + j
            phi_plus[row_idx, col_idx] = 1.0
        end
    end
    
    # 2. Apply the channel to the first subsystem
    # The channel acts on 'dim' x 'dim'. The reference is identity.
    # Since we can't easily tensor a function, we apply it by linearity.
    # Choi Matrix C = sum_ij N(|i><j|) ⊗ |i><j|
    
    choi_matrix = zeros(ComplexF64, d2, d2)
    
    # Basis matrices |i><j|
    basis_mats = [zeros(ComplexF64, dim, dim) for _ in 1:d2]
    
    for i in 1:dim
        for j in 1:dim
            # Create |i><j|
            E_ij = zeros(ComplexF64, dim, dim)
            E_ij[i, j] = 1.0
            
            # Apply the black box channel
            mapped_block = channel_func(E_ij)
            
            # Place this block into the Choi matrix
            # The Choi matrix structure is block_matrix C_ij = N(E_ij) ??
            # Standard definition: C = sum N(|i><j|) ⊗ |i><j|
            # This effectively puts mapped_block into the (i,j) block of the larger matrix
            
            row_start = (i - 1) * dim + 1
            row_end   = i * dim
            col_start = (j - 1) * dim + 1
            col_end   = j * dim
            
            choi_matrix[row_start:row_end, col_start:col_end] = mapped_block
        end
    end
    
    # 3. Diagonalize the Choi Matrix to find Kraus Ops
    # C = sum_k λ_k |v_k><v_k|
    # Kraus operator K_k = sqrt(λ_k) * reshape(v_k)
    
    # Ensure Hermitian for stability (Choi matrix must be positive semi-definite)
    choi_matrix = Hermitian(choi_matrix)
    vals, vecs = eigen(choi_matrix)
    
    kraus_ops = Vector{Matrix{ComplexF64}}()
    
    for k in 1:d2
        val = vals[k]
        
        # Filter out zero eigenvalues (numerical noise)
        if val > threshold
            # Extract eigenvector
            v = vecs[:, k]
            
            # Reshape vector into matrix (column-major is standard in Julia)
            # We need to be careful with the reshaping convention consistent with the Choi def.
            # If C = sum N(E_ij) ⊗ E_ij, then K = reshape(v) works directly if N acting on left.
            
            op = reshape(v, dim, dim)
            
            # Scale by eigenvalue
            K = sqrt(val) * op
            push!(kraus_ops, K)
        end
    end
    
    return kraus_ops
end