"""
    fidelity(ρ, σ; tol)

Compute quantum fidelity between density matrices ρ and σ.
"""
function fidelity(ρ, σ; tol=1e-10)
    ρ_sqrt = matrix_power_pseudo(ρ, 0.5, tol=tol)
    
    # Compute √ρ · σ · √ρ
    M = ρ_sqrt * σ * ρ_sqrt
    
    # Take square root of M
    M_sqrt = matrix_power_pseudo(M, 0.5, tol=tol)
    
    # Fidelity is [Tr(M_sqrt)]²
    F = abs(tr(M_sqrt))^2
    
    return real(F)  # Should be real, but numerical errors might give tiny imaginary part
end
