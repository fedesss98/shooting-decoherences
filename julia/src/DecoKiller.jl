module DecoKiller

using ProgressMeter

export 
PetzMaps,
get_dephasing_operators, get_amplitudedamping_operators, get_bitflip_operators, 
# initial configuration objects
load_configuration, RecoveryConfig, RecoveryState, RecoveryLogs, NoiseObj,
# initial states
codespace_state, thermal_state, random_state,
apply_channel, recovery_map,
# metrics
fidelity, overlap,

UnitaryDilation,
PetzCollisionModel, apply_petz_collision, extract_kraus_operators


include("PetzMaps.jl")
include("UnitaryDilation/UnitaryDilation.jl")


using .PetzMaps
using .UnitaryDilation

include("utils.jl")
include("configurations.jl")
# Add log functionality, printing to both console and file with timestamps and log levels
include("logging.jl")
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
        end

        # Setup the progress bar for one state evolution
        p_time = Progress(
            cfg.n_timesteps, desc=" State $s ", offset=1, keep=false)

        for _ in 1:cfg.n_timesteps
            step_recovery!(initial_state, cfg, logs)
            
            next!(p_time; showvalues=[
                ("Fidelity", logs.fidelities[end]),
                ("Reference", logs.ref_fidelities[end])
            ])
        end
        ref_fidelities .+= logs.ref_fidelities
        fidelities .+= logs.fidelities
        next!(p_states)
    end

end


end  # module
