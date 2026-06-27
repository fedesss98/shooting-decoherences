using LinearAlgebra

function basis_index(bits::AbstractVector{Int})
    idx = 0
    for b in bits
        idx = 2idx + b
    end
    return idx + 1
end

function ghz_state(n::Int)
    n >= 2 || throw(ArgumentError("n must be at least 2"))

    D = 2^n
    ψ = zeros(ComplexF64, D)

    ψ[1] = 1 / sqrt(2)      # |00...0⟩
    ψ[end] = 1 / sqrt(2)    # |11...1⟩

    return ψ
end

function w_state(n::Int)
    n >= 2 || throw(ArgumentError("n must be at least 2"))

    D = 2^n
    ψ = zeros(ComplexF64, D)

    for k in 1:n
        bits = zeros(Int, n)
        bits[k] = 1
        ψ[basis_index(bits)] = 1 / sqrt(n)
    end

    return ψ
end

function singlet_state()
    return ComplexF64[0, 1, -1, 0] / sqrt(2)
end

function kron_all_vec(states::Vector{Vector{ComplexF64}})
    out = states[1]
    for k in 2:length(states)
        out = kron(out, states[k])
    end
    return out
end

function singlet_pairs_state(n::Int)
    n >= 2 || throw(ArgumentError("n must be at least 2"))
    iseven(n) || throw(ArgumentError("singlet_pairs_state requires even n"))

    pairs = [singlet_state() for _ in 1:(n ÷ 2)]
    return kron_all_vec(pairs)
end

function noisy_entangled_dm(
    n::Int,
    p::Real;
    kind::Symbol = :ghz,
    ψ::Union{Nothing, AbstractVector} = nothing,
)
    n >= 2 || throw(ArgumentError("n must be at least 2"))
    0 <= p <= 1 || throw(ArgumentError("p must satisfy 0 <= p <= 1"))

    D = 2^n

    if ψ !== nothing
        length(ψ) == D || throw(ArgumentError("custom ψ must have length 2^n = $D"))
        ψref = ComplexF64.(ψ)
        ψref = ψref / norm(ψref)
    elseif kind == :ghz
        ψref = ghz_state(n)
    elseif kind == :w
        ψref = w_state(n)
    elseif kind == :singlet_pairs
        ψref = singlet_pairs_state(n)
    else
        throw(ArgumentError("unknown kind: $kind. Use :ghz, :w, :singlet_pairs, or pass ψ."))
    end

    ρpure = ψref * ψref'
    ρmix = Matrix{ComplexF64}(I, D, D) / D

    ρ = Float64(p) * ρpure + (1 - Float64(p)) * ρmix
    ρ = Matrix{ComplexF64}((ρ + ρ') / 2)
    ρ = ρ / real(tr(ρ))

    return ρ
end