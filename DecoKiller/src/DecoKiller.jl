module DecoKiller

using ProgressMeter
using JSON
using JLD2
using Dates

export 
run_experiment, 
iterate_recovery!,
kraus_to_superop,
get_kraus_operators,
vec, unvec, ptrace_sys, ptrace_anc,
fidelity, overlap, tr


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

include("experiments.jl")


end  # module
