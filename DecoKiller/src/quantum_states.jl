"""
    rand_unitary_haar(N)

Generates a random unitary matrix of size N×N distributed according 
to the Haar measure (CUE). 
Reference: F. Mezzadri, "How to generate random matrices from the classical compact groups"
"""
function rand_unitary_haar(N::Int, rng)
    # 1. Generate a random complex Gaussian matrix
    if !isnothing(rng)
        Z = randn(rng, ComplexF64, N, N) / sqrt(2.0)
    else
        Z = randn(ComplexF64, N, N) / sqrt(2.0)
    end

    # 2. Perform QR decomposition
    Q, R = qr(Z)

    # 3. Correct the phases of the diagonal of R to ensure strict Haar distribution
    # (The standard QR is unique only up to phases; this fixes the gauge)
    d = diag(R)
    phases = d ./ abs.(d)
    return Matrix(Q) * Diagonal(phases)
end

"""
    rand_state_with_spectrum(spectrum)

Generates a random Hermitian density matrix (or Hamiltonian) with the exact 
eigenvalues provided in `spectrum`.
"""
function rand_state_with_spectrum(spectrum::Vector{<:Number}; rng=nothing)
    # Ensure the spectrum is normalized
    spectrum = abs.(spectrum) / sum(abs.(spectrum))
    N = length(spectrum)

    # Generate the random basis change
    U = rand_unitary_haar(N, rng)

    # Construct the matrix: rho = U * Λ * U†
    # We use Diagonal for efficiency
    Lambda = Diagonal(spectrum)

    state = U * Lambda * U'
    for i in eachindex(state)
        re = abs(real(state[i])) > 1e-8 ? real(state[i]) : 0.0
        img = abs(imag(state[i])) > 1e-8 ? imag(state[i]) : 0.0
        state[i] = re + im * img
    end
    return state
end


"""
    fullrank_state(n_qubits)
Create a full-rank density matrix for n_qubits
"""
function fullrank_state(n_qubits)
    Random.seed!(1234)
    coeffs = normalize(randn(Float64, 2^n_qubits))
    r = zeros(ComplexF64, 2^n_qubits, 2^n_qubits)
    for (i, c) in enumerate(coeffs)
        psi = zeros(ComplexF64, 2^n_qubits)
        psi[i] = c
        r += psi * psi'
    end
    return r
end

"""
    thermal_state(n, beta)
Create the thermal state used to initialize the Petz map
"""
function thermal_state(n, beta)
    σz = [1.0+0.0im 0; 0 -1]
    q = zeros(ComplexF64, 2^n, 2^n)
    for i in 1:n
        for j in 1:i-1
            σz_i = foldl(kron, [k == i ? σz : I(2) for k in 1:n])
            σz_j = foldl(kron, [k == j ? σz : I(2) for k in 1:n])
            q += σz_i * σz_j
        end
    end
    return exp(beta * q) / tr(exp(beta * q))
end
function site_operator(op, site, n)
    id = Matrix{ComplexF64}(I, 2, 2)
    ops = [k == site ? op : id for k in 1:n]
    return foldl(kron, ops)
end

"""
    plus_zz_state(n_qubits; t=π / 4)

Create the pure state obtained by evolving `|+>^n` with the all-to-all
Ising unitary `exp(-im * t * sum_{i<j} Z_i Z_j)`.
For `n_qubits == 2`, this is exactly `exp(-im * Z ⊗ Z * t)|++>`.
"""
function plus_zz_state(n_qubits::Int; t::Real=π / 4)
    n_qubits >= 1 || throw(ArgumentError("n_qubits must be at least 1"))

    plus = ComplexF64[1, 1] / sqrt(2)
    ψ = foldl(kron, [plus for _ in 1:n_qubits])

    if n_qubits == 1
        return ψ
    end

    σz = ComplexF64[1 0; 0 -1]
    dim = 2^n_qubits
    H = zeros(ComplexF64, dim, dim)
    σz_ops = [site_operator(σz, i, n_qubits) for i in 1:n_qubits]

    for i in 1:n_qubits-1
        for j in i+1:n_qubits
            H += σz_ops[i] * σz_ops[j]
        end
    end

    return exp(-im * Float64(t) * H) * ψ
end

"""
    plus_zz_dm(n_qubits; t=π / 4)

Create the density matrix associated with [`plus_zz_state`](@ref).
"""
function plus_zz_dm(n_qubits::Int; t::Real=π / 4)
    ψ = plus_zz_state(n_qubits; t=t)
    return ψ * ψ'
end

"""
    singlet_state()

Create the two-qubit singlet state `|ψ-> = (|01> - |10>) / sqrt(2)`.
"""
function singlet_state()
    return ComplexF64[0, 1, -1, 0] / sqrt(2)
end

"""
    werner_dm(n_qubits, p)

Create the two-qubit Werner state
`p |ψ-><ψ-| + (1 - p) I / 4`, with `1/3 < p <= 1`.
"""
function werner_dm(n_qubits::Int, p::Real)
    n_qubits == 2 || throw(ArgumentError("werner reference state requires n_qubits = 2"))
    1 / 3 < p <= 1 || throw(ArgumentError("werner reference state requires 1/3 < p <= 1"))

    ψ = singlet_state()
    return Float64(p) * (ψ * ψ') + (1 - Float64(p)) * Matrix{ComplexF64}(I, 4, 4) / 4
end

function thermal_state_hopping(n, beta; g=1 / n)
    σp = ComplexF64[0 1; 0 0]
    σm = ComplexF64[0 0; 1 0]

    dim = 2^n
    H = zeros(ComplexF64, dim, dim)

    σp_ops = [site_operator(σp, i, n) for i in 1:n]
    σm_ops = [site_operator(σm, i, n) for i in 1:n]

    for i in 1:n-1
        for j in i+1:n
            H += g * (σp_ops[i] * σm_ops[j] + σp_ops[j] * σm_ops[i])
        end
    end

    ρ = exp(-beta * H)
    return ρ / tr(ρ)
end

"""
    input_state(n, a, b)

Create a random input state for n qubits of the form a|00...0> + b|11...1>
"""
function input_state(n, a, b)
    # Ground and excited states of one qubit
    g0 = [0.0 + 0.0im; 1.0]
    e1 = [1.0 + 0.0im; 0.0]

    ground = foldl(kron, [g0 for _ in 1:n])
    excited = foldl(kron, [e1 for _ in 1:n])
    state = normalize(a * ground + b * excited)
    return state
end

function _normalize_state(psi::Vector{ComplexF64})
    norm(psi) > 0 || throw(ArgumentError("state amplitudes cannot all be zero"))
    return normalize(psi)
end

function _random_codespace_amplitudes(rng::AbstractRNG)
    p = rand(rng)
    phase = 2π * rand(rng)
    return sqrt(p), sqrt(1 - p) * exp(-im * phase)
end

function _coherence_scale(coherence::Real)
    0 <= coherence <= 1 || throw(ArgumentError("coherence must be in [0, 1]"))
    return Float64(coherence)
end

function _coherence_scale(rng::Union{Nothing,AbstractRNG}, pure::Bool, coherence)
    coherence !== nothing && return _coherence_scale(coherence)
    pure && return 1.0
    return rng === nothing ? 0.0 : rand(rng)
end

function _damped_codespace_dm(psi::Vector{ComplexF64}, i::Int, j::Int, coherence::Real)
    rho = psi * psi'
    scale = _coherence_scale(coherence)
    rho[i, j] *= scale
    rho[j, i] *= scale
    return rho
end

"""
    codespace_state(n_qubits, alpha0, alpha1)
    codespace_state(n_qubits, rng)

Create a logic qubit `alpha0|00...0> + alpha1|11...1>`, where
`|00...0>` is at index 1 and `|11...1>` is at index `2^n_qubits`.
When an RNG is provided instead of amplitudes, the normalized amplitudes
are sampled randomly.
"""
function codespace_state(n_qubits, alpha0, alpha1)
    psi = zeros(ComplexF64, 2^n_qubits)
    psi[1] = alpha0
    psi[end] = alpha1
    return _normalize_state(psi)
end

function codespace_state(n_qubits, rng::AbstractRNG)
    alpha0, alpha1 = _random_codespace_amplitudes(rng)
    return codespace_state(n_qubits, alpha0, alpha1)
end

"""
    single_excitation_state(n_qubits, alpha0, alpha1)
    single_excitation_state(n_qubits, rng)

Create a state in the kernel of the all-to-all hopping Hamiltonian
"""
function single_excitation_state(n_qubits, alpha0, alpha1)
    n_qubits != 2 &&
        throw(ArgumentError("The parameter `codespace_xy` is implemented only for `n_qubits=2`."))
    psi = zeros(ComplexF64, 2^n_qubits)
    psi[2] = alpha0
    psi[3] = alpha1
    return _normalize_state(psi)
end

function single_excitation_state(n_qubits, rng::AbstractRNG)
    alpha0, alpha1 = _random_codespace_amplitudes(rng)
    return single_excitation_state(n_qubits, alpha0, alpha1)
end

"""
    codespace_dm(n_qubits, alpha0, alpha1; pure=true, coherence=nothing)
    codespace_dm(n_qubits, rng; pure=true, coherence=nothing)

Create a density matrix supported on `|00...0>` and `|11...1>`.
By default this is the pure-state outer product. Set `coherence < 1`, or
use `pure=false` with an RNG, to keep the same populations while reducing
the off-diagonal coherences.
"""
function codespace_dm(n_qubits, alpha0, alpha1; pure::Bool=true, coherence=nothing)
    psi = codespace_state(n_qubits, alpha0, alpha1)
    scale = _coherence_scale(nothing, pure, coherence)
    return _damped_codespace_dm(psi, 1, length(psi), scale)
end

function codespace_dm(n_qubits, rng::AbstractRNG; pure::Bool=true, coherence=nothing)
    psi = codespace_state(n_qubits, rng)
    scale = _coherence_scale(rng, pure, coherence)
    return _damped_codespace_dm(psi, 1, length(psi), scale)
end

function single_excitation_dm(n_qubits, alpha0, alpha1; pure::Bool=true, coherence=nothing)
    psi = single_excitation_state(n_qubits, alpha0, alpha1)
    scale = _coherence_scale(nothing, pure, coherence)
    return _damped_codespace_dm(psi, 2, 3, scale)
end

function single_excitation_dm(n_qubits, rng::AbstractRNG; pure::Bool=true, coherence=nothing)
    psi = single_excitation_state(n_qubits, rng)
    scale = _coherence_scale(rng, pure, coherence)
    return _damped_codespace_dm(psi, 2, 3, scale)
end

"""
    random_state(n_qubits)
Sample uniformly in the sphere
"""
function random_state(n_qubits; seed=nothing)
    rng = isnothing(seed) ? Random.default_rng() : Xoshiro(seed)
    ϕ = 2 * rand(rng) * π
    z = 2 * rand(rng) - 1

    θ = acos(z)

    α = cos(θ / 2)
    β = exp(im * ϕ) * sin(θ / 2)

    return input_state(n_qubits, α, β)
end

"""
    maxmix_state(n_qubits)
Returns the maximally mixed state for n_qubits
"""
function maxmix_state(n_qubits)
    return Matrix(I, 2^n_qubits, 2^n_qubits) / 2^n_qubits
end
