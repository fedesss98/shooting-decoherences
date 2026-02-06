
function partial_traces(ρ_total, d_s, d_a)
    
    ρ_s = zeros(ComplexF64, d_s, d_s)
    ρ_a = zeros(ComplexF64, d_a, d_a)
    
    # For kron(ρ, η), index = (i-1)*d_a + k
    # where i ∈ {1,...,d_s} is system, k ∈ {1,...,d_a} is ancilla
    
    # Trace out ancilla
    for i in 1:d_s
        for j in 1:d_s
            for k in 1:d_a
                idx_row = (i-1)*d_a + k
                idx_col = (j-1)*d_a + k
                ρ_s[i, j] += ρ_total[idx_row, idx_col]
            end
        end
    end
    
    # Trace out system
    for k1 in 1:d_a
        for k2 in 1:d_a
            for i in 1:d_s
                idx_row = (i-1)*d_a + k1
                idx_col = (i-1)*d_a + k2
                ρ_a[k1, k2] += ρ_total[idx_row, idx_col]
            end
        end
    end
    
    return ρ_s, ρ_a
end
