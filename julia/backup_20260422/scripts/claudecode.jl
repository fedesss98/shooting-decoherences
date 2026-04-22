using LinearAlgebra

function petz_recovery_stinespring(K::Vector{Matrix{ComplexF64}}, ρ0::Matrix{ComplexF64})
    d = size(K[1], 1)
    m = length(K)
    
    # Forward channel
    ρ1 = sum(k * ρ0 * k' for k in K)
    
    # Petz recovery Kraus operators
    ρ0_sqrt = sqrt(Hermitian(ρ0))
    ρ1_inv_sqrt = inv(sqrt(Hermitian(ρ1 + 1e-10*I)))
    M = [ρ0_sqrt * k' * ρ1_inv_sqrt for k in K]
    
    # Stinespring isometry V: d → d⊗m
    # V = sum_i M_i ⊗ |i⟩⟨0|
    V = zeros(ComplexF64, d*m, d)
    for i in 1:m
        V[(i-1)*d+1:i*d, :] = M[i]
    end
    
    # Recovery: ρ_rec = V ρ1 V†
    ρ_rec = V' * kron(ρ1, Matrix(1.0I, m, m)) * V
    
    # Simpler: direct application
    ρ_rec = sum(M[i] * ρ1 * M[i]' for i in 1:m)
    
    return ρ_rec, V, M, ρ1
end

# Example
function example()
  d = 2
  γ = 0.3
  K = [
      ComplexF64[1 0; 0 sqrt(1-γ)],
      ComplexF64[0 sqrt(γ); 0 0]
  ]

  ρ0 = ComplexF64[0.7 0.1; 0.1 0.3]
  ρ_rec, V, M, ρ1 = petz_recovery_stinespring(K, ρ0)

  fid(ρ, σ) = real(tr(sqrt(sqrt(Hermitian(ρ)) * σ * sqrt(Hermitian(ρ)))))

  println("Initial state:\n", ρ0)
  println("\nAfter channel:\n", ρ1)
  println("Fidelity with initial: ", fid(ρ0, ρ1))
  println("\nRecovered state:\n", ρ_rec)
  println("Fidelity with initial: ", fid(ρ0, ρ_rec))
end