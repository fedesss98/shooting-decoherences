module DecoKiller

using ProgressMeter

export 
PetzMaps,
get_kraus_operators, 
# initial configuration objects
load_configuration, RecoveryConfig, RecoveryState, RecoveryLogs, NoiseObj, ChoiceSystem,
# initial states
codespace_state, codespace_dm, thermal_state, random_state, input_state,
apply_channel, recovery_map, rand_state_with_spectrum,
# metrics
fidelity, overlap,
partial_traces,
UnitaryDilation,
CollisionModel, apply_collision, extract_kraus_operators,
# main function
run_experiment, discrimin, update_noise_history!, measure_ancilla, update_noise_guess, step_recovery!

include("PetzMaps.jl")
include("UnitaryDilation/UnitaryDilation.jl")


using .PetzMaps
using .UnitaryDilation

include("utils.jl")
# Add log functionality, printing to both console and file with timestamps and log levels
include("logging.jl")
# Add configuration files loading and setup
include("configurations.jl")
# Implement the main algorithm
include("adaptive_recovery.jl")


function run_experiment(config_file="./configs/config.toml")
    cfg, state, logs = load_configuration(config_file)
    
    ref_fidelities = zeros(Float64, cfg.n_timesteps)
    fidelities = zeros(Float64, cfg.n_timesteps)

    p_states = Progress(cfg.n_states, desc="Adaptive Recovery ")
    for s in 1:cfg.n_states
        initial_state = deepcopy(state)  # Reset state for each run
        if cfg.recovery_type == "random"
            ψ0 = random_state(cfg.n_qubits)
            ρ0 = ψ0 * ψ0'
            initial_state.ρ0 .= ρ0
            initial_state.ρ_free .= ρ0
            initial_state.ρ_rec .= ρ0
        elseif cfg.recovery_type == "codespace"
            a, b, c, d = rand(cfg.rng, 4)
            ψ0 = codespace_state(cfg.n_qubits, a, b, c, d)
            ρ0 = ψ0 * ψ0'
            initial_state.ρ0 .= ρ0
            initial_state.ρ_free .= ρ0
            initial_state.ρ_rec .= ρ0
        elseif cfg.recovery_type == "inputstate"
            a, b = rand(cfg.rng, 2)
            ψ0 = input_state(cfg.n_qubits, a, b)
            ρ0 = ψ0 * ψ0'
            initial_state.ρ0 .= ρ0
            initial_state.ρ_free .= ρ0
            initial_state.ρ_rec .= ρ0
        end

        # Setup the progress bar for one state evolution
        p_time = Progress(
            cfg.n_timesteps, desc=" State $s ", offset=1)

        for step in 1:cfg.n_timesteps
            step_recovery!(step, initial_state, cfg, logs)
            
            next!(p_time; showvalues=[
                ("Fidelity", logs.fidelities[end]),
                ("Reference", logs.ref_fidelities[end])
            ])
        end
        finish!(p_time)
        start_state_results = (s - 1) * cfg.n_timesteps + 1
        end_state_results = s * cfg.n_timesteps

        ref_fidelities .+= logs.ref_fidelities[start_state_results:end_state_results]
        fidelities .+= logs.fidelities[start_state_results:end_state_results]
        next!(p_states)
    end
    avg_fidelities = ref_fidelities ./ cfg.n_states, fidelities ./ cfg.n_states
    return cfg, state, logs, avg_fidelities
end


end  # module
