#!/usr/bin/env julia
# =============================================================================
#
# Reads  configs/config_iter.toml,
# builds every cartesian combination of the parameters listed under [iterate]
# (ignoring `base_name`), then for each combination:
#
#   1. Creates  experiments/<base_name><suffix>/config.toml
#      (flat TOML – no [iterate] section, no [settings] header; [plots] kept)
#   2. Calls    DecoKiller.run_experiment(config_path)
#   3. Extracts the last avg fidelity *with* recovery: avg_fidelities[2][end]
#   4. Appends a row to  experiments/iterations_summary.csv
#
# Combination suffix format: n{n_qubits}_b{beta_no_dot}
#   e.g.  n_qubits=2, beta=0.1  →  "n2_b01"
#         n_qubits=3, beta=1.0  →  "n3_b10"
#
# CSV column order mirrors the standard config.toml field order (no [plots]).
# noise_options is expanded in-place as noise_N_prob / noise_N_name / noise_N_gamma.
# =============================================================================

# ── Activate the Julia project and load DecoKiller ────────────────────────────
# Script is at  julia/scripts/run_iterations.jl
# Project.toml is at  julia/Project.toml  →  one level up from @__DIR__
const SCRIPT_DIR = @__DIR__

import Pkg
Pkg.activate(joinpath(SCRIPT_DIR, ".."); io=devnull)   # silent activation

include(joinpath("..", "src", "DecoKiller.jl"))
using .DecoKiller                                        # exports run_experiment
using TOML

# ── Project-level paths ───────────────────────────────────────────────────────
const PROJECT_ROOT = normpath(joinpath(SCRIPT_DIR, "..", ".."))
const CONFIG_ITER_PATH = joinpath(PROJECT_ROOT, "configs", "config_iter.toml")
const EXPERIMENTS_DIR = joinpath(PROJECT_ROOT, "experiments")
const SUMMARY_CSV = joinpath(EXPERIMENTS_DIR, "iterations_summary.csv")

# ── CSV column order: mirrors standard config.toml, [plots] excluded ──────────
# noise_options is expanded at this position as noise_N_prob/name/gamma columns.
const CSV_FIELDS_PRE_NOISE = [
    "name",
    "n_qubits", "beta",
    "dt", "n_timesteps", "n_states",
    "starting_state", "recovery_type",
    "collision_unitary", "ancilla_state", "ancilla_dim", "ancilla_alpha",
    "seed",
]
const CSV_FIELDS_POST_NOISE = [
    "real_noise", "correlated_noise",
]


# =============================================================================
# Helpers
# =============================================================================

function parse_cli_args(args::Vector{String})
    config_path = CONFIG_ITER_PATH
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

"""
    fmt_no_dot(v)  →  String

Strip the decimal separator from a number's string representation.
  0.1  →  "01"   |   1.0  →  "10"   |   2  →  "2"
"""
fmt_no_dot(v::Real) = replace(string(v), "." => "")
fmt_no_dot(v::Integer) = string(v)

"""
    combo_suffix(combo)  →  String

Build the folder-name suffix for one parameter combination.
`n_qubits` is always prefixed with 'n', `beta` with 'b'; any additional
iterated parameter uses its key as prefix (alphabetical order).
"""
function combo_suffix(combo::AbstractDict)::String
    parts = String[]
    haskey(combo, "n_qubits") && push!(parts, "n$(combo["n_qubits"])")
    haskey(combo, "beta") && push!(parts, "b$(fmt_no_dot(combo["beta"]))")
    for k in sort(collect(keys(combo)))
        k ∈ ("n_qubits", "beta") && continue
        push!(parts, "$(k)$(fmt_no_dot(combo[k]))")
    end
    return join(parts, "_")
end

"""
    noise_col_names(noise_options)  →  Vector{String}

Return the CSV column names for a noise_options list:
  ["noise_1_prob", "noise_1_name", "noise_1_gamma", "noise_2_prob", …]
"""
function noise_col_names(noise_options::AbstractVector)::Vector{String}
    cols = String[]
    for i in eachindex(noise_options)
        push!(cols, "noise_$(i)_prob", "noise_$(i)_name", "noise_$(i)_gamma")
    end
    return cols
end

"""
    expand_noise_options(noise_options)  →  Dict{String,Any}

Flatten a noise_options list into individual CSV-ready key/value pairs:
  [[0.5, "amplitude_damping", 10.0], …]
  →  Dict("noise_1_prob"=>0.5, "noise_1_name"=>"amplitude_damping", "noise_1_gamma"=>10.0, …)
"""
function expand_noise_options(noise_options::AbstractVector)::Dict{String,Any}
    d = Dict{String,Any}()
    for (i, entry) in enumerate(noise_options)
        d["noise_$(i)_prob"] = entry[1]
        d["noise_$(i)_name"] = entry[2]
        d["noise_$(i)_gamma"] = entry[3]
    end
    return d
end

"""
    csv_cell(v)  →  String

Serialise a value as a CSV cell, RFC-4180 quoting when necessary.
"""
function csv_cell(v)::String
    s = v isa AbstractArray ? repr(v) : string(v)
    if any(c -> c ∈ (',', '\n', '"'), s)
        return "\"$(replace(s, "\"" => "\"\""))\""
    end
    return s
end

"""
    append_csv_row(path, row, col_order)

Append one data row to `path`.  The header is written first when the file
does not yet exist.  Missing keys are written as empty cells.
"""
function append_csv_row(path::String, row::AbstractDict, col_order::Vector{String})
    write_header = !isfile(path)
    open(path, "a") do io
        write_header && println(io, join(col_order, ","))
        println(io, join([csv_cell(get(row, c, "")) for c in col_order], ","))
    end
end


# =============================================================================
# Main
# =============================================================================

function main(args=ARGS)
    mkpath(EXPERIMENTS_DIR)

    # ── 1. Load config_iter.toml ───────────────────────────────────────────────
    config_path, debug = parse_cli_args(args)
    cfg = TOML.parsefile(config_path)
    iterate_cfg = cfg["iterate"]
    settings_cfg = cfg["settings"]
    plots_cfg = get(cfg, "plots", Dict{String,Any}())

    # ── 2. Separate base_name from the parameters to iterate over ─────────────
    base_name = iterate_cfg["base_name"]
    iter_params = Dict(k => v for (k, v) in iterate_cfg if k != "base_name")

    param_names = sort(collect(keys(iter_params)))   # stable order
    param_vals = [iter_params[k] for k in param_names]

    # ── 3. Build the CSV column order ─────────────────────────────────────────
    # Fixed pre/post-noise fields + dynamically expanded noise columns + result
    noise_opts = get(settings_cfg, "noise_options", [])
    noise_cols = noise_col_names(noise_opts)
    col_order = vcat(CSV_FIELDS_PRE_NOISE, noise_cols, CSV_FIELDS_POST_NOISE, ["avg_fidelity"])

    n_total = prod(length.(param_vals))
    println("Starting $(n_total) experiment(s) …\n")

    # ── 4. Iterate over the full cartesian product ─────────────────────────────
    for combo_vals in Iterators.product(param_vals...)

        combo = Dict(param_names[i] => combo_vals[i] for i in eachindex(param_names))

        # 4a. Resolve folder name and paths
        suffix = combo_suffix(combo)
        folder_name = base_name * suffix
        folder_path = joinpath(EXPERIMENTS_DIR, folder_name)
        config_path = joinpath(folder_path, "config.toml")
        mkpath(folder_path)

        println("┌─ $folder_name")

        # 4b. Write config.toml
        #     Flat layout: name + iter params + all [settings] values + [plots] table.
        #     No [iterate] section; no [settings] header.
        flat_cfg = Dict{String,Any}("name" => folder_name)
        merge!(flat_cfg, combo)         # n_qubits, beta (and any extras)
        merge!(flat_cfg, settings_cfg)  # dt, n_timesteps, noise_options, …
        isempty(plots_cfg) || (flat_cfg["plots"] = plots_cfg)

        open(config_path, "w") do io
            TOML.print(io, flat_cfg)
        end
        println("│  config      → $config_path")

        # 4c. Run the experiment
        flat_cfg["n_states"] == 1 && @warn "n_states == 1: per-state fidelity averages unavailable; recording NaN."

        avg_fidelities = (nothing, nothing)
        try
            _, _, _, avg_fidelities = run_experiment(config_path)
        catch err
            @warn "Error while running simulation" exception = (err, catch_backtrace())
            continue
        end

        avg_fidelity = if isnothing(avg_fidelities[2])
            NaN
        else
            avg_fidelities[2][end]
        end

        println("└  Average fidelity ($(flat_cfg["n_timesteps"]) steps) = $avg_fidelity\n")

        # 4d. Build CSV row
        #     All settings added individually; noise_options replaced by expanded columns.
        row = Dict{String,Any}("name" => folder_name, "avg_fidelity" => avg_fidelity)
        merge!(row, combo)
        for (k, v) in settings_cfg
            k == "noise_options" && continue   # expanded separately below
            row[k] = v
        end
        merge!(row, expand_noise_options(noise_opts))

        append_csv_row(SUMMARY_CSV, row, col_order)
    end

    println("✓  All $(n_total) experiment(s) complete.")
    println("   Summary → $SUMMARY_CSV")
end

main()
