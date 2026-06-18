using Test
using LinearAlgebra
using Random

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

@testset "codespace constructors" begin
    ψ = codespace_state(3, 3.0, 4.0im)

    @test length(ψ) == 8
    @test norm(ψ) ≈ 1
    @test ψ[1] ≈ 3 / 5
    @test ψ[end] ≈ 4im / 5
    @test all(iszero, ψ[2:end-1])

    ρ = codespace_dm(3, 3.0, 4.0im)
    @test ρ ≈ ψ * ψ'
    @test tr(ρ) ≈ 1
    @test ρ * ρ ≈ ρ

    rng_state = Xoshiro(42)
    rng_dm = Xoshiro(42)
    ψ_random = codespace_state(3, rng_state)
    ρ_random = codespace_dm(3, rng_dm)

    @test norm(ψ_random) ≈ 1
    @test ρ_random ≈ ψ_random * ψ_random'
    @test tr(ρ_random) ≈ 1
    @test ρ_random * ρ_random ≈ ρ_random

    ψ_xy = single_excitation_state(2, Xoshiro(7))
    ρ_xy = single_excitation_dm(2, Xoshiro(7))

    @test norm(ψ_xy) ≈ 1
    @test ψ_xy[1] == 0
    @test ψ_xy[4] == 0
    @test ρ_xy ≈ ψ_xy * ψ_xy'
    @test_throws ArgumentError codespace_state(2, 0, 0)
    @test_throws ArgumentError single_excitation_state(3, 1, 0)
end
