using TOML
using Random
using StatsBase

# Operators relative to the noise
mutable struct NoiseObj
  name::String
  probability::Float64
  kraus::Vector{Matrix{ComplexF64}}
  supermap_petz::Matrix{ComplexF64}
  supermap_noise::Matrix{ComplexF64}
  supermap::Matrix{ComplexF64}
end
function NoiseObj(noise::String, p::Float64, sigma::Matrix{T}, gamma::Float64, t::Float64) where T
  kraus = get_kraus_operators(noise, gamma, t)
  n_qubits = Int(log2(size(sigma, 1)))
  model = CollisionModel(kraus, sigma, n=n_qubits)
  M_petz, M_noise = build_superoperators(model)
  return NoiseObj(noise, p, kraus, M_petz, M_noise, M_noise)
end

# Static Configuration for the setup of the algorithm
struct RecoveryConfig
    name::String
    sigma::Matrix{ComplexF64}
    recovery_type::String
    real_noise::NoiseObj
    n_qubits::Int
    n_timesteps::Int
    n_states::Int
    seed::Int
    rng::AbstractRNG
    dt::Float64
end

mutable struct ChoiceSystem
    c1::Vector{Int}
    c2::Vector{Int}
    c1_count::Int
    c2_count::Int
    current::Int
    current_choices::Vector{Int}
end

# Dynamic State: Updates every iteration
mutable struct RecoveryState
    ρ0::Matrix{ComplexF64}
    ρ_free::Matrix{ComplexF64}
    ρ_rec::Matrix{ComplexF64}
    noise_guess::NoiseObj
    M_total::Matrix{ComplexF64}
    choice::ChoiceSystem
    noise_options::Vector{NoiseObj}
end

# Define the type for logging the supermaps at each step (using a NamedTuple for clarity)
const SupermapsLogged = @NamedTuple{Nx::Matrix{ComplexF64}, N1::Matrix{ComplexF64}, N2::Matrix{ComplexF64}, P::Matrix{ComplexF64}}
# Logs for tracking evolution of metrics
struct RecoveryLogs
    fidelities::Vector{Float64}
    ref_fidelities::Vector{Float64}
    # This will store NamedTuples containing 4 matrices
    maps::Vector{SupermapsLogged}
end
# Constructor to initialize empty logs
RecoveryLogs() = RecoveryLogs(Float64[], Float64[], SupermapsLogged[])

"""
    load_configuration(config_file)

Reads a TOML configuration file and initializes all components needed
to run the recovery algorithm: configuration, initial state, and logs.
"""
function load_configuration(config_file="./configs/config.toml")
    cfg = TOML.parsefile(config_file)
    recovery_cfg  = parse_recovery_config(cfg)
    noise_options  = parse_noise_options(cfg, recovery_cfg.sigma, recovery_cfg.dt)
    recovery_state = initialize_recovery_state(recovery_cfg, noise_options)
    return recovery_cfg, recovery_state, RecoveryLogs()
end

# Helper function to create RNG from seed, allowing for reproducibility or randomness
make_rng(seed) = seed == -1 ? Random.default_rng() : Xoshiro(seed)


function parse_recovery_config(cfg::Dict)::RecoveryConfig
    name          = get(cfg, "name",          "test")
    n_qubits      = get(cfg, "n_qubits",      1)
    beta          = get(cfg, "beta",          2.0)
    dt            = get(cfg, "dt",            0.1)
    n_timesteps   = get(cfg, "n_timesteps",   10)
    n_states      = get(cfg, "n_states",      1)
    seed          = get(cfg, "seed",          42)
    recovery_type = get(cfg, "recovery_type", "auto")
    starting_state = get(cfg, "starting_state", "thermal")

    rng           = make_rng(seed)
    experiment_dir = setup_experiment_dir(name, cfg)
    setup_logger(joinpath(experiment_dir, "debug.log"))

    sigma         = make_reference_state(starting_state, n_qubits, beta, rng)
    noise_options = parse_noise_options(cfg, sigma, dt)
    if get(cfg, "real_noise", nothing) === nothing
        println("No real noise specified in config, sampling from noise options...")
        real_noise = sample_real_noise(rng, noise_options)
    else
        println("Using specified real noise from config...")
        real_noise = [n for n in noise_options if n.name == cfg["real_noise"]][1]
    end

    return RecoveryConfig(
        name, sigma, recovery_type, real_noise,
        n_qubits, n_timesteps, n_states, seed, rng, dt
    )
end


function setup_experiment_dir(name::String, cfg::Dict)::String
    root_dir       = dirname(dirname(@__DIR__))
    experiment_dir = joinpath(root_dir, "experiments", name)

    mkpath(joinpath(experiment_dir, "logs"))
    mkpath(joinpath(experiment_dir, "visualization"))

    open(joinpath(experiment_dir, "config.toml"), "w") do io
        TOML.print(io, cfg)
    end

    return experiment_dir
end

function make_reference_state(kind::String, n_qubits::Int, beta::Float64, rng)
    if kind == "thermal"
        return thermal_state(n_qubits, beta)
    elseif kind == "random"
        spectrum = 0.1 .+ 0.9 .* rand(rng, 2^n_qubits)
        return rand_state_with_spectrum(spectrum; rng=rng)
    else
        throw(ArgumentError("Unsupported starting state: $kind"))
    end
end

const DEFAULT_NOISE_OPTIONS = [
    (0.60, "bitflip",   1.0),
    (0.40, "dephasing", 1.0),
]

function parse_noise_options(cfg::Dict, sigma, dt::Float64)::Vector{NoiseObj}
    raw = get(cfg, "noise_options", DEFAULT_NOISE_OPTIONS)
    return [NoiseObj(name, prob, sigma, gamma, dt)
            for (prob, name, gamma) in raw]
end

function sample_real_noise(rng, noise_options::Vector{NoiseObj})::NoiseObj
    weights = Weights([n.probability for n in noise_options])
    return deepcopy(sample(rng, noise_options, weights))
end


function initialize_recovery_state(cfg::RecoveryConfig, noise_options::Vector{NoiseObj})::RecoveryState
    ρ0     = make_initial_state(cfg)
    choice = make_initial_choice(cfg.rng, noise_options)

    noise_guess = deepcopy(noise_options[choice.current])
    M_total     = noise_guess.supermap_noise

    return RecoveryState(
        copy(ρ0), copy(ρ0), copy(ρ0),
        noise_guess, M_total, choice, noise_options
    )
end

function make_initial_state(cfg::RecoveryConfig)
    if cfg.recovery_type == "random"
        ψ = random_state(cfg.n_qubits)
        return ψ * ψ'
    elseif cfg.recovery_type == "auto"
        return copy(cfg.sigma)
    elseif cfg.recovery_type == "codespace"
        sigma = cfg.sigma
        p = rand()
        max_x = p * (1-p)  # Maximum allowed magnitude for |x|^2
        radius = sqrt(rand() * max_x)
        x = radius * exp(2π * im * rand())
        ρ = codespace_dm(cfg.n_qubits, p, x)
        return ρ + sigma
    elseif cfg.recovery_type == "inputstate"
        a, b = rand(cfg.rng, 2)
        ψ = input_state(cfg.n_qubits, a, b)
        return ψ * ψ'
    else
        throw(ArgumentError("Unsupported recovery type: $(cfg.recovery_type)"))
    end
end

function make_initial_choice(rng, noise_options::Vector{NoiseObj})::ChoiceSystem
    n       = length(noise_options)
    counts  = zeros(Int, n)
    current = sample(rng, 1:n)
    counts[current] = 1

    # Build per-hypothesis count vectors: 1 for chosen, 0 for others
    count_vecs = [i == current ? [1] : [0] for i in 1:n]

    return ChoiceSystem(count_vecs..., counts..., current, [current])
end