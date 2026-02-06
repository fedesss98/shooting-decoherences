

function unitary_dilation(K_ops, d_s, d_a)
    """
    Apply a unitary dilation to system+ancilla that implements Kraus operators on the system.
    
    Arguments:
    - K_ops: vector of Kraus operators [K_0, K_1, ..., K_{n-1}]
    - d_s: dimension of system
    - d_a: dimension of ancilla (must be ≥ length(K_ops))
    
    The unitary acts as: U = Σ_k K_k ⊗ |k⟩⟨0|
    assuming the ancilla starts in |0⟩ state.
    
    Returns:
    - U: the dilation unitary matrix
    """
    
    n_kraus = length(K_ops)
    
    if d_a < n_kraus
        error("Ancilla dimension ($d_a) must be ≥ number of Kraus operators ($n_kraus)")
    end
    
    d_total = d_s * d_a
    U = zeros(ComplexF64, d_total, d_total)
    
    # Build unitary dilation: U|j⟩_s|0⟩_a = Σ_k K_k|j⟩_s ⊗ |k⟩_a
    # For kron(system, ancilla), index = (i-1)*d_a + k
    
    for i in 1:d_s
        for j in 1:d_s
            # Input state: |j⟩_s|0⟩_a
            idx_in = (j-1)*d_a + 1
            
            # Apply each Kraus operator
            for (k_idx, K) in enumerate(K_ops)
                # Output: K_k|j⟩_s ⊗ |k-1⟩_a (k_idx starts at 1, so ancilla state is k_idx-1)
                ancilla_state = k_idx  # This is the ancilla index (1-indexed)
                idx_out = (i-1)*d_a + ancilla_state
                U[idx_out, idx_in] = K[i, j]
            end
        end
    end
    
    # For ancilla states |k⟩ with k > 0, we need to complete the unitary
    # Apply identity on these subspaces
    for k in 2:d_a
        for i in 1:d_s
            idx = (i-1)*d_a + k
            if U[idx, idx] == 0.0  # Only if not already filled
                U[idx, idx] = 1.0
            end
        end
    end
    
    return U
end