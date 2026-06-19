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

    ρ_damped = codespace_dm(3, 3.0, 4.0im; coherence=0.5)
    @test diag(ρ_damped) ≈ diag(ρ)
    @test ρ_damped[1, end] ≈ 0.5 * ρ[1, end]
    @test ρ_damped[end, 1] ≈ 0.5 * ρ[end, 1]
    @test real(tr(ρ_damped * ρ_damped)) < 1
    @test minimum(eigvals(Hermitian(ρ_damped))) >= -1e-12

    rng_state = Xoshiro(42)
    rng_dm = Xoshiro(42)
    ψ_random = codespace_state(3, rng_state)
    ρ_random = codespace_dm(3, rng_dm)

    @test norm(ψ_random) ≈ 1
    @test ρ_random ≈ ψ_random * ψ_random'
    @test tr(ρ_random) ≈ 1
    @test ρ_random * ρ_random ≈ ρ_random

    rng_mixed_state = Xoshiro(99)
    rng_mixed_dm = Xoshiro(99)
    ψ_mixed = codespace_state(3, rng_mixed_state)
    coherence = rand(rng_mixed_state)
    ρ_mixed_codespace = codespace_dm(3, rng_mixed_dm; pure=false)

    @test diag(ρ_mixed_codespace) ≈ diag(ψ_mixed * ψ_mixed')
    @test ρ_mixed_codespace[1, end] ≈ coherence * ψ_mixed[1] * conj(ψ_mixed[end])
    @test real(tr(ρ_mixed_codespace * ρ_mixed_codespace)) < 1

    ψ_xy = single_excitation_state(2, Xoshiro(7))
    ρ_xy = single_excitation_dm(2, Xoshiro(7))

    @test norm(ψ_xy) ≈ 1
    @test ψ_xy[1] == 0
    @test ψ_xy[4] == 0
    @test ρ_xy ≈ ψ_xy * ψ_xy'
    @test single_excitation_dm(2, 3.0, 4.0; coherence=0.25)[2, 3] ≈ 0.25 * 12 / 25
    @test_throws ArgumentError codespace_state(2, 0, 0)
    @test_throws ArgumentError single_excitation_state(3, 1, 0)
    @test_throws ArgumentError codespace_dm(2, 1, 1; coherence=1.1)
end

@testset "plus ZZ reference state" begin
    t = π / 4
    ψ = plus_zz_state(2; t=t)
    expected = ComplexF64[
        exp(-im * t),
        exp(im * t),
        exp(im * t),
        exp(-im * t),
    ] / 2

    @test ψ ≈ expected
    @test norm(ψ) ≈ 1

    ρ = plus_zz_dm(2; t=t)
    @test ρ ≈ expected * expected'
    @test tr(ρ) ≈ 1
    @test ρ * ρ ≈ ρ

    ρ3 = plus_zz_dm(3)
    @test size(ρ3) == (8, 8)
    @test tr(ρ3) ≈ 1
    @test ρ3 * ρ3 ≈ ρ3
    @test minimum(eigvals(Hermitian(ρ3))) >= -1e-12
    @test_throws ArgumentError plus_zz_state(0)
end

include("../src/DecoKiller.jl")

@testset "pinned collapse maps" begin
    ρ = Matrix{ComplexF64}(I, 4, 4) / 4
    zero_target = zeros(ComplexF64, 4, 4)

    @test Main.DecoKiller.collapse_map(ρ, zero_target; pin=true) == zeros(ComplexF64, 16, 16)
    @test_throws ArgumentError PetzMaps.kraus_to_superop(Matrix{ComplexF64}[])
end
