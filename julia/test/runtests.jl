using Test
using LinearAlgebra

include("../src/PetzMaps.jl")
using .PetzMaps

@testset "fidelity" begin
    σ = ComplexF64[
        0.2 0.01 0.0 0.05
        0.01 0.3 0.0 0.0
        0.0 0.0 0.1 0.0
        0.05 0.0 0.0 0.4
    ]
    σ = (σ + σ') / 2
    σ ./= tr(σ)

    p = 0.3
    ψ = ComplexF64[sqrt(p), 0, 0, sqrt(1 - p)]
    ρ_pure = ψ * ψ'
    ρ_mixed = ComplexF64[
        p 0 0 0
        0 0 0 0
        0 0 0 0
        0 0 0 1 - p
    ]

    @test fidelity(ρ_pure, σ) ≈ real(tr(ρ_pure * σ))
    @test fidelity(σ, ρ_pure) ≈ real(tr(ρ_pure * σ))
    @test real(tr(ρ_mixed * ρ_mixed)) < 1
    @test fidelity(ρ_mixed, σ) != real(tr(ρ_mixed * σ))
end
