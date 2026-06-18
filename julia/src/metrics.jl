"""
    fidelity(ρ, σ; tol)

Compute quantum fidelity between density matrices ρ and σ.

This returns the squared Uhlmann fidelity,

    F(ρ, σ) = (tr(sqrt(sqrt(ρ) * σ * sqrt(ρ))))^2.

If either argument is a pure density matrix, this reduces to
`real(tr(ρ * σ))`. A diagonal state like
`α|00><00| + β|11><11|` is generally mixed; the pure coherent state is
`(a|00> + b|11>)(a|00> + b|11>)'`, including the off-diagonal terms.
"""
function fidelity(ρ::AbstractMatrix, σ::AbstractMatrix; tol=1e-10)
    size(ρ) == size(σ) || throw(DimensionMismatch("ρ and σ must have the same size"))
    size(ρ, 1) == size(ρ, 2) || throw(DimensionMismatch("ρ and σ must be square matrices"))

    ρ_sqrt = matrix_power_pseudo(ρ, 0.5, tol=tol)

    # Compute √ρ · σ · √ρ
    M = ρ_sqrt * σ * ρ_sqrt

    # Take square root of M
    M_sqrt = matrix_power_pseudo(M, 0.5, tol=tol)

    # Fidelity is [Tr(M_sqrt)]²
    F = abs(tr(M_sqrt))^2

    return clamp_fidelity(real(F), tol)
end


function overlap(ρ, σ::AbstractMatrix)
    return real(tr(ρ * σ))
end


function overlap(ρ, ψ::AbstractVector)
    return real(ψ' * ρ * ψ)
end


function clamp_fidelity(F::Real, tol)
    F < 0 && F >= -tol && return 0.0
    F > 1 && F <= 1 + tol && return 1.0
    return Float64(F)
end
