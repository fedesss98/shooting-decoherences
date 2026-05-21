using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "DecoKiller.jl"))
using .DecoKiller

function main()
    config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "configs", "config.toml")
    cfg, _, logs, avg_fids = run_experiment(config_path)

    println("Experiment completed")
    println("Name: $(cfg.name)")
    println("Output: $(cfg.experiment_dir)")
    println("Steps: $(cfg.n_timesteps)")
    println("Final fidelity (with recovery): $(round(logs.fidelities[end], digits=6))")
    println("Final fidelity (no recovery):  $(round(logs.ref_fidelities[end], digits=6))")
    println("Average final fidelity (with recovery): $(round(avg_fids[2][end], digits=6))")
    println("Average final fidelity (no recovery):  $(round(avg_fids[1][end], digits=6))")
end

main()
