using TOML
using Random
using StatsBase

# Operators relative to the noise
mutable struct NoiseObj{T<:Number}
  name::String
  probability::Float64
  kraus::Vector{Matrix{T}}
  supermap_petz::Matrix{T}
  supermap_noise::Matrix{T}
  supermap::Matrix{T}
end
function NoiseObj(noise::String, p::Float64, sigma::Matrix{T}, gamma::Float64, t::Float64) where T
  kraus = get_kraus_operators(noise, gamma, t)
  n_qubits = Int(log2(size(sigma, 1)))
  model = CollisionModel(kraus, sigma, n=n_qubits)
  M_petz, M_noise = build_superoperators(model)
  return NoiseObj{T}(noise, p, kraus, M_petz, M_noise, M_noise)
end

# Static Configuration for the setup of the algorithm
struct RecoveryConfig{T<:Number}
    sigma::Matrix{T}
    recovery_type::String
    real_noise::NoiseObj
    n_qubits::Int
    n_timesteps::Int
    n_states::Int
    seed::Int
    rng::AbstractRNG
end

# Dynamic State: Updates every iteration
mutable struct RecoveryState{T<:Number}
    ρ0::Matrix{T}
    ρ_free::Matrix{T}
    ρ_rec::Matrix{T}
    noise_guess::NoiseObj
    M_total::Matrix{T}
    c1::Int
    c2::Int
    noise_options::Vector{NoiseObj{T}}
end

# Logs for tracking evolution of metrics
struct RecoveryLogs
    fidelities::Vector{Float64}
    ref_fidelities::Vector{Float64}
end
# Constructor to initialize empty logs
RecoveryLogs() = RecoveryLogs(Float64[], Float64[])


"""
    load_configuration(config_file)
Reads a TOML configuration file to initialize the setup of the algorithm
"""
function load_configuration(config_file="./configs/config.toml")
    cfg = TOML.parsefile(config_file)

    n_qubits = get(cfg, "n_qubits", 1)
    beta = get(cfg, "beta", 2.0)
    gamma = get(cfg, "gamma", 1.0)
    dt = get(cfg, "dt", 0.1)
    n_timesteps = get(cfg, "n_timesteps", 10)
    seed = get(cfg, "seed", 42)
    recovery_type = get(cfg, "type", "auto")
    starting_state = get(cfg, "starting_state", "thermal")
    n_states = get(cfg, "n_states", 1)

    c1 = c2 = 0

    # Create the reference state for the recovery
    if starting_state == "thermal"
        sigma = thermal_state(n_qubits, beta)
    elseif starting_state == "random"
        ψ = random_state(n_qubits; seed=seed)
        sigma = ψ * ψ'
    else
        error("Unsupported starting state: $starting_state")
    end

    if recovery_type == "iterative"
        ψ = random_state(n_qubits; seed=seed)
        ρ0 = ψ * ψ'
    elseif recovery_type == "auto"
        ρ0 = copy(sigma)
    end

    # Generate the possible noises and precompute their operators
    default_noises = [
        (0.60, "bitflip"),
        (0.40, "dephasing"),
    ]
    noise_probabilities = get(cfg, "noise_probabilities", default_noises)
    noise_options = [
        NoiseObj(noise_model[2], noise_model[1], sigma, gamma, dt)
        for noise_model in noise_probabilities
    ]
    # Initialize Random Number Generator with the seed
    rng = Xoshiro(seed)

    # Choose randomly a noise model
    real_noise = sample(rng,
        [n[2] for n in noise_probabilities], 
        Weights([n[1] for n in noise_probabilities])
    )
    real_noise = NoiseObj(real_noise, 1.0, sigma, gamma, dt)

    # Take an initial guess for the noise model
    noise_guess = sample(rng, [n for n in noise_options])

    recovery_cfg = RecoveryConfig(
        sigma, 
        recovery_type, 
        real_noise, 
        n_qubits, 
        n_timesteps, 
        n_states, 
        seed,
        rng
    )

    
    M_petz, M_noise = noise_guess.supermap_petz, noise_guess.supermap_noise
    M_total = M_noise

    recovery_state = RecoveryState(
       copy(ρ0),
       copy(ρ0),
       copy(ρ0),
       noise_guess, 
       M_total, 
       c1, c2, 
       noise_options
    )

    logs = RecoveryLogs()

    return recovery_cfg, recovery_state, logs

end