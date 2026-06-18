using TOML
using Random
using StatsBase

# Operators relative to the noise
mutable struct NoiseObj
    name::String
    probability::Float64
    kraus::Vector{Matrix{ComplexF64}}
    extended_kraus::Vector{Matrix{ComplexF64}}
    supermap_petz::Matrix{ComplexF64}
    supermap_noise::Matrix{ComplexF64}
    supermap::Matrix{ComplexF64}
end
function NoiseObj(
    noise::String,
    p::Float64,
    sigma::Matrix{T},
    gamma::Float64,
    t::Float64;
    rng::Union{Nothing,AbstractRNG}=nothing,
    correlated::Bool=false
) where T
    n_qubits = Int(log2(size(sigma, 1)))

    # kraus is always the single-qubit channel. extended_kraus is the
    # full-system channel used by the recovery protocol.
    kraus = _make_noise_kraus(noise, gamma, t; rng=rng, n_qubits=1)
    extended_kraus =
        correlated && n_qubits > 1 ? _make_correlated_kraus(noise, gamma, t, kraus; rng=rng, n_qubits=n_qubits) :
        expand_kraus_operators(kraus, n_qubits)

    model = CollisionModel(extended_kraus, sigma)
    M_petz, M_noise = build_superoperators(model)
    return NoiseObj(noise, p, kraus, extended_kraus, M_petz, M_noise, M_noise)
end

function _make_correlated_kraus(noise::String, gamma::Float64, t::Float64, kraus; rng=nothing, n_qubits::Int)
    if occursin("random", noise) || occursin("general", noise)
        return correlate_kraus_operators(kraus, n_qubits)
    end
    return _make_noise_kraus(noise, gamma, t; rng=rng, n_qubits=n_qubits)
end

function _make_noise_kraus(noise::String, gamma::Float64, t::Float64; rng=nothing, n_qubits::Int=1)
    if occursin("random", noise) || occursin("general", noise)
        rng === nothing && throw(ArgumentError("Noise type $noise requires a RNG."))
        return get_kraus_operators(noise, gamma, t; rng=rng, n_qubits=n_qubits)
    end
    return get_kraus_operators(noise, gamma, t; n_qubits=n_qubits)
end

# Static Configuration for the setup of the algorithm
struct RecoveryConfig
    name::String
    experiment_dir::String
    sigma::Matrix{ComplexF64}
    sigma_mixture::Float64
    recovery_type::String
    real_noise::NoiseObj
    n_qubits::Int
    n_timesteps::Int
    n_states::Int
    seed::Int
    rng::AbstractRNG
    dt::Float64
    ancilla_alpha::Float64
    ancilla_dim::Int
    ancilla_state_name::String
    collision_unitary_name::String
    ancilla_state::Matrix{ComplexF64}
    collision_unitary::Matrix{ComplexF64}
    correlated_noise::Bool
    plots_options::Dict{Symbol,Any}
    noise_options::Vector{NoiseObj}
    real_noise_idx::Int
    codespace_projection::String
    pin::Bool
end

function RecoveryConfig(
    name::String,
    experiment_dir::String,
    sigma::Matrix{ComplexF64},
    sigma_mixture::Float64,
    recovery_type::String,
    real_noise::NoiseObj,
    n_qubits::Int,
    n_timesteps::Int,
    n_states::Int,
    seed::Int,
    rng::AbstractRNG,
    dt::Float64,
    ancilla_alpha::Float64;
    ancilla_state_name::String="thermal_qubit",
    ancilla_dim::Int=default_ancilla_dim(ancilla_state_name),
    collision_unitary_name::String="swap",
    correlated_noise::Bool=false,
    plots_options::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    noise_options::Union{Nothing,Vector{NoiseObj}}=nothing,
    real_noise_idx::Int=0,
    codespace_projection::String="auto",
    pin::Bool=false
)
    ancilla_state = make_ancilla_state(ancilla_state_name, ancilla_alpha, ancilla_dim)
    collision_unitary = make_collision_unitary(
        collision_unitary_name,
        2^n_qubits,
        size(ancilla_state, 1),
    )

    return RecoveryConfig(
        name, experiment_dir, sigma, sigma_mixture, recovery_type, real_noise,
        n_qubits, n_timesteps, n_states, seed, rng, dt, ancilla_alpha, ancilla_dim,
        ancilla_state_name, collision_unitary_name, ancilla_state, collision_unitary,
        correlated_noise, plots_options, isnothing(noise_options) ? NoiseObj[real_noise] : noise_options,
        real_noise_idx, codespace_projection, pin
    )
end

mutable struct ChoiceSystem
    c1::Vector{Int}
    c2::Vector{Int}
    c1_count::Int
    c2_count::Int
    current::Int
    history::Vector{Int}
end

# Dynamic State: Updates every iteration
mutable struct RecoveryState
    ρ0::Matrix{ComplexF64}
    ρ_free::Matrix{ComplexF64}
    ρ_rec::Matrix{ComplexF64}
    ρ_codespace0::Matrix{ComplexF64}
    noise_guess::NoiseObj
    M_total::Matrix{ComplexF64}
    choice::ChoiceSystem
    noise_options::Vector{NoiseObj}
end

# Define the type for logging the supermaps at each step (using a NamedTuple for clarity)
const SupermapsLogged = @NamedTuple{
    Nx::Matrix{ComplexF64}, N1::Matrix{ComplexF64}, N2::Matrix{ComplexF64},
    P::Matrix{ComplexF64}, Cx::Matrix{ComplexF64}, Xi::Matrix{ComplexF64}}
# Logs for tracking evolution of metrics
struct RecoveryLogs
    fidelities::Vector{Float64}
    ref_fidelities::Vector{Float64}
    codespace_overlaps::Vector{Vector{Float64}}
    choice_history::Vector{Int}
    # This will store NamedTuples containing 4 matrices
    maps::Vector{SupermapsLogged}
end
# Constructor to initialize empty logs
RecoveryLogs() = RecoveryLogs(Float64[], Float64[], [Float64[], Float64[], Float64[]], Int[], SupermapsLogged[])

"""
    load_configuration(config_file)

Reads a TOML configuration file and initializes all components needed
to run the recovery algorithm: configuration, initial state, and logs.
"""
function load_configuration(config_file="./configs/config.toml"; debug::Bool=false)
    cfg = TOML.parsefile(config_file)
    recovery_cfg = parse_recovery_config(cfg; debug=debug)
    noise_options = recovery_cfg.noise_options
    @debug "Parsed Noise Options: " [no.name for no in noise_options]
    recovery_state = initialize_recovery_state(recovery_cfg, noise_options)
    return recovery_cfg, recovery_state, RecoveryLogs()
end

# Helper function to create RNG from seed, allowing for reproducibility or randomness
make_rng(seed) = seed == -1 ? Random.default_rng() : Xoshiro(seed)

default_ancilla_dim(kind::String)::Int = kind == "ground_qudit" ? 4 : 2

function make_ancilla_state(kind::String, alpha::Float64, dim::Int=default_ancilla_dim(kind))::Matrix{ComplexF64}
    dim >= 2 || throw(ArgumentError("ancilla_dim must be at least 2"))
    if kind == "thermal_qubit"
        dim == 2 || throw(ArgumentError("thermal_qubit requires ancilla_dim = 2"))
        return Matrix{ComplexF64}(ancilla_thermal_qubit(alpha; T=ComplexF64))
    elseif kind == "ground_qubit"
        dim == 2 || throw(ArgumentError("ground_qubit requires ancilla_dim = 2; use ground_qudit for larger ancillae"))
        return Matrix{ComplexF64}(ancilla_ground_state(ComplexF64, 2))
    elseif kind == "ground_qudit"
        return Matrix{ComplexF64}(ancilla_ground_state(ComplexF64, dim))
    elseif kind == "mixture"
        dim == 2 || throw(ArgumentError("mixture requires ancilla_dim = 2"))
        η = [[alpha 0]; [0 1 - alpha]]
        return Matrix{ComplexF64}(η)
    else
        throw(ArgumentError("Unsupported ancilla_state: $kind"))
    end
end


function make_collision_unitary(kind::String, ds::Int, da::Int)::Matrix{ComplexF64}
    if kind == "swap"
        return _swap_unitary(ds, da)
    elseif kind == "jc"
        da == 2 || throw(ArgumentError("collision_unitary = \"jc\" requires ancilla_dim = 2"))
        n_qubits = Int(log2(ds))
        return _n_qubit_exchange_unitary(n_qubits)
    else
        throw(ArgumentError("Unsupported collision_unitary: $kind"))
    end
end


function parse_recovery_config(cfg::Dict; debug::Bool=false)::RecoveryConfig
    name = get(cfg, "name", "test")
    n_qubits = get(cfg, "n_qubits", 1)
    beta = get(cfg, "beta", 2.0)
    dt = get(cfg, "dt", 0.1)
    anc_alpha = get(cfg, "ancilla_alpha", 0.8)
    anc_type = get(cfg, "ancilla_state", "thermal_qubit")
    ancilla_dim = get(cfg, "ancilla_dim", default_ancilla_dim(anc_type))
    collision_type = get(cfg, "collision_unitary", "swap")
    n_timesteps = get(cfg, "n_timesteps", 10)
    n_states = get(cfg, "n_states", 1)
    seed = get(cfg, "seed", 42)
    recovery_type = get(cfg, "recovery_type", "auto")
    starting_state = get(cfg, "starting_state", "thermal")
    sigma_mixture = get(cfg, "sigma_mixture", 0.5)
    correlated_noise = get(cfg, "correlated_noise", false)
    codespace_projection = get(cfg, "codespace_projection", "auto")
    pin = get(cfg, "pin", false)
    plots_options = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in get(cfg, "plots", Dict{String,Any}()))

    rng = make_rng(seed)
    experiment_dir = setup_experiment_dir(name, cfg)
    setup_logger(joinpath(experiment_dir, "debug.log"); console_level=debug ? Logging.Debug : Logging.Info)

    sigma = make_reference_state(starting_state, n_qubits, beta, rng)
    noise_options = parse_noise_options(cfg, sigma, dt, rng)
    if get(cfg, "real_noise", nothing) === nothing
        println("No real noise specified in config, sampling from noise options...\n")
        real_noise_idx = sample_real_noise_index(rng, noise_options)
    else
        println("Using specified real noise from config: $(cfg["real_noise"])\n")
        real_noise_idx = findfirst(n -> n.name == cfg["real_noise"], noise_options)
        if real_noise_idx === nothing
            throw(ArgumentError("Specified real noise '$(cfg["real_noise"])' not found in noise options"))
        end
    end
    real_noise = deepcopy(noise_options[real_noise_idx])

    return RecoveryConfig(
        name, experiment_dir, sigma, sigma_mixture, recovery_type, real_noise,
        n_qubits, n_timesteps, n_states, seed, rng, dt, anc_alpha;
        ancilla_dim=ancilla_dim,
        ancilla_state_name=anc_type,
        collision_unitary_name=collision_type,
        correlated_noise=correlated_noise,
        plots_options=plots_options,
        noise_options=noise_options,
        real_noise_idx=real_noise_idx,
        codespace_projection=codespace_projection,
        pin=pin
    )
end


function setup_experiment_dir(name::String, cfg::Dict)::String
    root_dir = dirname(dirname(@__DIR__))
    experiment_dir = joinpath(root_dir, "experiments", name)

    mkpath(joinpath(experiment_dir, "logs"))
    mkpath(joinpath(experiment_dir, "visualization"))
    mkpath(joinpath(experiment_dir, "data"))

    open(joinpath(experiment_dir, "config.toml"), "w") do io
        TOML.print(io, cfg)
    end

    return experiment_dir
end

function make_reference_state(kind::String, n_qubits::Int, beta::Float64, rng)
    if kind == "thermal"
        return thermal_state(n_qubits, beta)
    elseif kind == "thermal_xy"
        return thermal_state_hopping(n_qubits, beta)
    elseif kind == "random"
        spectrum = 0.1 .+ 0.9 .* rand(rng, 2^n_qubits)
        return rand_state_with_spectrum(spectrum; rng=rng)
    elseif kind == "codespace"
        return codespace_dm(n_qubits, 0.6, 0.4)
    else
        throw(ArgumentError("Unsupported starting state: $kind"))
    end
end

function make_reference_state(params::Dict, n_qubits::Int, beta::Float64, rng)
    n_qubits > 1 && throw(ArgumentError("Input state with angles is only supported for single qubit systems"))
    if haskey(params, "rx") && haskey(params, "ry") && haskey(params, "rz")
        rx = params["rx"]
        ry = params["ry"]
        rz = params["rz"]
        ρ = 0.5 * [
            [1 + rz rx - im * ry];
            [rx + ry * im 1 - rz]
        ]
        return ρ
    elseif haskey(params, "rx2") && haskey(params, "ry2") && haskey(params, "rz2")
        rx = sqrt(params["rx2"])
        ry = sqrt(params["ry2"])
        rz = sqrt(params["rz2"])
        ρ = 0.5 * [
            [1 + rz rx - im * ry];
            [rx + ry * im 1 - rz]
        ]
        return ρ
    else
        throw(ArgumentError("Invalid starting_state parameters. Expected keys: 'rx', 'ry', 'rz' or 'rx2', 'ry2', 'rz2'"))
    end
end

const DEFAULT_NOISE_OPTIONS = [
    (0.60, "bitflip", 1.0),
    (0.40, "dephasing", 1.0),
]

function parse_noise_options(cfg::Dict, sigma, dt::Float64, rng::Union{AbstractRNG,Nothing}=nothing)::Vector{NoiseObj}
    raw = get(cfg, "noise_options", DEFAULT_NOISE_OPTIONS)
    correlated_noise = get(cfg, "correlated_noise", false)
    return [NoiseObj(name, prob, sigma, gamma, dt; rng=rng, correlated=correlated_noise)
            for (prob, name, gamma) in raw]
end

function sample_real_noise(rng, noise_options::Vector{NoiseObj})::NoiseObj
    return deepcopy(noise_options[sample_real_noise_index(rng, noise_options)])
end

function sample_real_noise_index(rng, noise_options::Vector{NoiseObj})::Int
    weights = Weights([n.probability for n in noise_options])
    return sample(rng, 1:length(noise_options), weights)
end


function initialize_recovery_state(cfg::RecoveryConfig, noise_options::Vector{NoiseObj})::RecoveryState
    ρ0, ρ_codespace0 = make_initial_state_and_reference(cfg)
    choice = make_initial_choice(cfg.rng, noise_options)

    noise_guess = deepcopy(noise_options[choice.current])
    M_total = noise_guess.supermap_noise

    return RecoveryState(
        copy(ρ0), copy(ρ0), copy(ρ0), copy(ρ_codespace0),
        noise_guess, M_total, choice, noise_options
    )
end

function make_initial_state(cfg::RecoveryConfig)
    ρ0, _ = make_initial_state_and_reference(cfg)
    return ρ0
end

function make_initial_state_and_reference(cfg::RecoveryConfig)
    if cfg.recovery_type == "random"
        ψ = random_state(cfg.n_qubits)
        ρ0 = ψ * ψ'
        return ρ0, ρ0
    elseif cfg.recovery_type == "auto"
        ρ0 = copy(cfg.sigma)
        return ρ0, ρ0
    elseif cfg.recovery_type == "codespace" || cfg.recovery_type == "codespace_xy"
        sigma = cfg.sigma
        p = rand()
        max_x = p * (1 - p)  # Maximum allowed magnitude for |x|^2
        radius = sqrt(rand() * max_x)
        x = radius * exp(2π * im * rand())
        ρ = if cfg.recovery_type == "codespace"
            codespace_dm(cfg.n_qubits, cfg.rng)
        else
            single_excitation_dm(cfg.n_qubits, p, x)
        end
        r = cfg.sigma_mixture
        ρ0 = (1 - r) * ρ + r * sigma
        ρ0 = ρ0 / tr(ρ0)  # Normalize to ensure it's a valid density matrix
        return ρ0, ρ
    elseif cfg.recovery_type == "inputstate"
        a, b = rand(cfg.rng, 2)
        ψ = input_state(cfg.n_qubits, a, b)
        ρ0 = ψ * ψ'
        return ρ0, ρ0
    else
        throw(ArgumentError("Unsupported recovery type: $(cfg.recovery_type)"))
    end
end

function make_initial_choice(rng, noise_options::Vector{NoiseObj})::ChoiceSystem
    n = length(noise_options)
    counts = zeros(Int, n)
    current = sample(rng, 1:n)
    counts[current] = 1

    # Build per-hypothesis count vectors: 1 for chosen, 0 for others
    count_vecs = [i == current ? [1] : [0] for i in 1:n]

    return ChoiceSystem(count_vecs..., counts..., current, [current])
end
