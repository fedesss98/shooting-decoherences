module DecoKiller

using ProgressMeter
using JSON
using Dates

export run_experiment, iterate_recovery!

# Core primitives and lower-level modules.
include("PetzMaps.jl")
include("UnitaryDilation/UnitaryDilation.jl")

using .PetzMaps
using .UnitaryDilation

# Shared utilities and configuration.
include("utils.jl")
include("logging.jl")
include("configurations.jl")

# Algorithm and reporting.
include("adaptive_recovery.jl")
include("custom_plots.jl")


function reset_initial_state!(state::RecoveryState, cfg::RecoveryConfig)
    if cfg.recovery_type == "random"
        ψ0 = random_state(cfg.n_qubits)
        ρ0 = ψ0 * ψ0'
        state.ρ0 .= ρ0
        state.ρ_free .= ρ0
        state.ρ_rec .= ρ0
    elseif cfg.recovery_type == "codespace" || cfg.recovery_type == "codespace_xy"
        sigma = cfg.sigma
        p = rand(cfg.rng)
        max_x = p * (1 - p)  # Maximum allowed magnitude for |x|^2
        radius = sqrt(rand(cfg.rng) * max_x)
        x = radius * exp(2π * im * rand(cfg.rng))
        ρ = if cfg.recovery_type == "codespace"
            codespace_dm(cfg.n_qubits, p, x)
        else
            single_excitation_dm(cfg.n_qubits, p, x)
        end
        r = cfg.sigma_mixture
        ρ0 = (1 - r) * ρ + r * sigma
        ρ0 = ρ0 / tr(ρ0)  # Normalize to ensure it's a valid density matrix
        state.ρ0 .= ρ0
        state.ρ_free .= ρ0
        state.ρ_rec .= ρ0
    elseif cfg.recovery_type == "inputstate"
        a, b = rand(cfg.rng, 2)
        ψ0 = input_state(cfg.n_qubits, a, b)
        ρ0 = ψ0 * ψ0'
        state.ρ0 .= ρ0
        state.ρ_free .= ρ0
        state.ρ_rec .= ρ0
    end
end


function _run_single_state!(cfg::RecoveryConfig, state::RecoveryState, logs::RecoveryLogs, s::Int)
    initial_state = deepcopy(state)
    reset_initial_state!(initial_state, cfg)

    p_time = Progress(cfg.n_timesteps, desc=" State $s ", offset=1)
    for step in 1:cfg.n_timesteps
        iterate_recovery!(step, initial_state, cfg, logs)
        next!(p_time; showvalues=[
            ("Fidelity", logs.fidelities[end]),
            ("Reference", logs.ref_fidelities[end])
        ])
    end
    finish!(p_time)

    start_idx = (s - 1) * cfg.n_timesteps + 1
    end_idx = s * cfg.n_timesteps
    return start_idx, end_idx
end


function _plot_and_save_results(cfg::RecoveryConfig, state::RecoveryState, logs::RecoveryLogs, avg_fidelities=nothing)
    save_results!(cfg, state, logs, avg_fidelities)
    if cfg.n_states == 1
        plot_autorecovery(state, cfg, logs; show=false, save=true, cfg.plots_options...)
    else
        plot_average_fidelity(
            avg_fidelities[2], avg_fidelities[1], state, cfg;
            show=false, save=true, cfg.plots_options...)
    end
    return nothing
end


function run_experiment(config_file="./configs/config.toml"; debug::Bool=false)
    cfg, state, logs = load_configuration(config_file; debug=debug)

    ref_fidelities = zeros(Float64, cfg.n_timesteps)
    fidelities = zeros(Float64, cfg.n_timesteps)

    if cfg.n_states == 1
        start_idx, end_idx = _run_single_state!(cfg, state, logs, 1)
        avg_fidelities = [nothing, nothing]
    else
    p_states = Progress(cfg.n_states, desc="Adaptive Recovery ")
    for s in 1:cfg.n_states
        start_idx, end_idx = _run_single_state!(cfg, state, logs, s)
        ref_fidelities .+= logs.ref_fidelities[start_idx:end_idx]
        fidelities .+= logs.fidelities[start_idx:end_idx]
        next!(p_states)
    end
    avg_fidelities = (ref_fidelities ./ cfg.n_states, fidelities ./ cfg.n_states)
    end
    _plot_and_save_results(cfg, state, logs, avg_fidelities)

    return cfg, state, logs, avg_fidelities
end

function _serialize_maps(logs::RecoveryLogs)
    return [
        Dict(
            "Nx" => vec(real(m.Nx)),
            "N1" => vec(real(m.N1)),
            "N2" => vec(real(m.N2)),
            "P" => vec(real(m.P)),
            "dim" => size(m.Nx)
        ) for m in logs.maps
    ]
end

function save_results!(cfg::RecoveryConfig, state::RecoveryState, logs::RecoveryLogs, avg_fidelities)
    out_file = joinpath(cfg.experiment_dir, "data", "results.json")
    payload = Dict(
        "timestamp" => string(now()),
        "experiment" => cfg.name,
        "n_qubits" => cfg.n_qubits,
        "n_timesteps" => cfg.n_timesteps,
        "n_states" => cfg.n_states,
        "seed" => cfg.seed,
        "real_noise" => cfg.real_noise.name,
        "noise_options" => [n.name for n in state.noise_options],
        "ancilla_state" => cfg.ancilla_state_name,
        "collision_unitary" => cfg.collision_unitary_name,
        "fidelities" => logs.fidelities,
        "ref_fidelities" => logs.ref_fidelities,
        "choice_history" => logs.choice_history,
        "choice_c1" => state.choice.c1,
        "choice_c2" => state.choice.c2,
        "avg_ref_fidelities" => avg_fidelities[1],
        "avg_fidelities" => avg_fidelities[2],
        "maps" => _serialize_maps(logs)
    )

    open(out_file, "w") do io
        JSON.print(io, payload, 2)
    end
    return out_file
end


end  # module
