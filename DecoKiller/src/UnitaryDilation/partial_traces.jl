"""
*How the indexing works.*
In Julia `kron(A, B)``, element `(i,j)` of `A` scales an entire copy of `B`, 
so the combined matrix has a block structure where block `[i,j]` equals `A[i,j] * B`.
That means:
  - to get `ρ_sys[i,j]`, sum the trace of each `(da×da)` block at position `[i,j]` 
  in the big matrix. This is the standard "sum diagonal blocks" approach.
  - to get `ρ_anc[k,l]`, pick the k,l element from each diagonal block `[i,i]`
   and sum over i. This visits the sys-diagonal elements that connect ancilla 
   index k to l.
"""


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



"""
    ptrace_ancilla(ρ, ds::Int, da::Int) -> Matrix

Partial trace over the ancilla subsystem from ρ = kron(sys, ancilla).
Returns the reduced density matrix of the system, size (ds × ds).

Convention: Julia's kron(A, B) = A ⊗ B, so 'sys' is the slow (outer) index.
Each (da × da) diagonal block [k,k] of ρ contributes to the trace.
"""
function ptrace_ancilla(ρ::AbstractMatrix, ds::Int, da::Int)
    @assert size(ρ) == (ds*da, ds*da) "ρ must be (ds·da)×(ds·da)"
    ρ_sys = zeros(eltype(ρ), ds, ds)
    for i in 1:ds
        for j in 1:ds
            # Block (i,j) of the sys structure: rows (i-1)*da+1:i*da, cols (j-1)*da+1:j*da
            block = ρ[(i-1)*da+1 : i*da, (j-1)*da+1 : j*da]
            ρ_sys[i, j] = tr(block)
        end
    end
    return ρ_sys / tr(ρ_sys)
end


"""
    ptrace_sys(ρ, ds::Int, da::Int) -> Matrix

Partial trace over the system subsystem from ρ = kron(sys, ancilla).
Returns the reduced density matrix of the ancilla, size (da × da).

Uses the reshape trick: ρ[i·da+k, j·da+l] contributes to ρ_anc[k,l]
when summed over i=j (diagonal in the sys space).
"""
function ptrace_sys(ρ::AbstractMatrix, ds::Int, da::Int)
    @assert size(ρ) == (ds*da, ds*da) "ρ must be (ds·da)×(ds·da)"
    ρ_anc = zeros(eltype(ρ), da, da)
    for k in 1:da
        for l in 1:da
            # Sum the sys-diagonal elements that couple ancilla indices k and l
            s = zero(eltype(ρ))
            for i in 1:ds
                s += ρ[(i-1)*da + k, (i-1)*da + l]
            end
            ρ_anc[k, l] = s
        end
    end
    return ρ_anc / tr(ρ_anc)
end