
function reset_initial_state!(state::RecoveryState, cfg::RecoveryConfig)
    if cfg.recovery_type == "random"
        ψ0 = random_state(cfg.n_qubits)
        ρ0 = ψ0 * ψ0'
        state.ρ0 .= ρ0
        state.ρ_free .= ρ0
        state.ρ_rec .= ρ0
        state.ρ_codespace0 .= ρ0
    elseif occursin("codespace", cfg.recovery_type)
        sigma = cfg.sigma
        ρ = if cfg.recovery_type == "codespace"
            codespace_dm(cfg.n_qubits, cfg.rng)
        elseif cfg.recovery_type == "codespace_xy"
            single_excitation_dm(cfg.n_qubits, cfg.rng)
        elseif cfg.recovery_type == "codespace_mxd"
            codespace_dm(cfg.n_qubits, cfg.rng; pure=false)
        else
            throw(ArgumentError("Unsupported recovery type: $(cfg.recovery_type)"))
        end
        r = cfg.sigma_mixture
        ρ0 = (1 - r) * ρ + r * sigma
        ρ0 = ρ0 / tr(ρ0)  # Normalize to ensure it's a valid density matrix
        state.ρ0 .= ρ0
        state.ρ_free .= ρ0
        state.ρ_rec .= ρ0
        state.ρ_codespace0 .= ρ
    elseif cfg.recovery_type == "inputstate"
        a, b = rand(cfg.rng, 2)
        ψ0 = input_state(cfg.n_qubits, a, b)
        ρ0 = ψ0 * ψ0'
        state.ρ0 .= ρ0
        state.ρ_free .= ρ0
        state.ρ_rec .= ρ0
        state.ρ_codespace0 .= ρ0
    end
end


function _run_single_state!(cfg::RecoveryConfig, state::RecoveryState, logs::RecoveryLogs, s::Int)
    initial_state = deepcopy(state)
    if s>1
        reset_initial_state!(initial_state, cfg)
    end

    @debug "Initial state ρ0: $(initial_state.ρ0)"

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
    return start_idx, end_idx, initial_state
end


function _plot_and_save_results(cfg::RecoveryConfig, state::RecoveryState, logs::RecoveryLogs, avg_fidelities=nothing)
    save_results!(cfg, state, logs, avg_fidelities)
    if cfg.n_states == 1
        @debug "Saving autorecovery results in directory: $(cfg.experiment_dir)"
        plot_autorecovery(state, cfg, logs; show=false, save=true, cfg.plots_options...)
    else
        @debug "Saving average fidelity results in directory: $(cfg.experiment_dir)"
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
        start_idx, end_idx, initial_state = _run_single_state!(cfg, state, logs, 1)
        state = deepcopy(initial_state)
        avg_fidelities = [nothing, nothing]
    else
        p_states = Progress(cfg.n_states, desc="Adaptive Recovery ")
        for s in 1:cfg.n_states
            start_idx, end_idx, _ = _run_single_state!(cfg, state, logs, s)
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
            "Cx" => vec(real(m.Cx)),
            "Xi" => vec(real(m.Xi)),
            "P" => vec(real(m.P)),
            "dim" => size(m.Nx)
        ) for m in logs.maps
    ]
end

function save_results!(cfg::RecoveryConfig, state::RecoveryState, logs::RecoveryLogs, avg_fidelities)
    meta_file = joinpath(cfg.experiment_dir, "data", "meta.json")
    payload = Dict(
        "timestamp" => string(now()),
        "experiment" => cfg.name,
        "n_qubits" => cfg.n_qubits,
        "n_timesteps" => cfg.n_timesteps,
        "n_states" => cfg.n_states,
        "seed" => cfg.seed,
        "real_noise" => cfg.real_noise.name,
        "real_noise_idx" => cfg.real_noise_idx,
        "noise_options" => [n.name for n in state.noise_options],
        "codespace_projection" => cfg.codespace_projection,
        "pin" => cfg.pin,
        "ancilla_dim" => cfg.ancilla_dim,
    )

    open(meta_file, "w") do io
        JSON.print(io, payload, 2)
    end

    matrices_file = joinpath(cfg.experiment_dir, "data", "superoperators.jld2")
    payload = Dict(
        "real_noise_kraus" => cfg.real_noise.extended_kraus,
        "real_noise_idx" => cfg.real_noise_idx,
        "noise_options_kraus" => [n.extended_kraus for n in state.noise_options],
        "ancilla_state" => cfg.ancilla_state_name,
        "ancilla_dim" => cfg.ancilla_dim,
        "collision_unitary" => cfg.collision_unitary_name,
        "maps" => _serialize_maps(logs)
    )
    JLD2.save(matrices_file, payload)

    results_file = joinpath(cfg.experiment_dir, "data", "results.json")
    payload = Dict(
        "fidelities" => logs.fidelities,
        "ref_fidelities" => logs.ref_fidelities,
        "avg_ref_fidelities" => avg_fidelities[1],
        "avg_fidelities" => avg_fidelities[2],
        "codespace_overlaps" => logs.codespace_overlaps,
        "choice_history" => logs.choice_history,
        "choice_c1" => state.choice.c1,
        "choice_c2" => state.choice.c2,
    )

    open(results_file, "w") do io
        JSON.print(io, payload, 2)
    end

    return results_file
end
