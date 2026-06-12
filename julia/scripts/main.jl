using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "DecoKiller.jl"))
using .DecoKiller


function parse_cli_args(args::Vector{String})
    config_path = joinpath(@__DIR__, "..", "..", "configs", "config.toml")
    debug = false

    for arg in args
        if arg == "--debug"
            debug = true
        elseif startswith(arg, "-")
            error("Unknown flag: $arg")
        else
            config_path = arg
        end
    end

    return config_path, debug
end

function main(args=ARGS)
    config_path, debug = parse_cli_args(args)
    cfg, _, logs, avg_fids = run_experiment(config_path; debug=debug)

    println("Experiment completed")
    println("Name: $(cfg.name)")
    println("Output: $(cfg.experiment_dir)")
    println("Steps: $(cfg.n_timesteps)")
    println("Final fidelity (with recovery): $(round(logs.fidelities[end], digits=6))")
    println("Final fidelity (no recovery):  $(round(logs.ref_fidelities[end], digits=6))")
    if cfg.n_states > 1
        println("Average final fidelity (with recovery): $(round(avg_fids[2][end], digits=6))")
        println("Average final fidelity (no recovery):  $(round(avg_fids[1][end], digits=6))")
    end
end

main()
