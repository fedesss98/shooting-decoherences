const SCRIPT_DIR = @__DIR__

import Pkg
Pkg.activate(joinpath(SCRIPT_DIR, ".."); io=devnull)   # silent activation

include(joinpath("..", "src", "DecoKiller.jl"))
using .DecoKiller                                        # exports run_experiment
using TOML
using CSV, DataFrames
using Plots
using LaTeXStrings

# --- Paths ----------------------------------------------
const ROOT_DIR = joinpath(SCRIPT_DIR, "..", "..")
const EXPERIMENTS_DIR = joinpath(ROOT_DIR, "experiments")
const SUMMARY_CSV = joinpath(EXPERIMENTS_DIR, "iterations_summary.csv")

# --- Utilities ------------------------------------------
const MARKER_SHAPES = [:circle, :utriangle, :dtriangle, :ltriangle, :rtriangle]

function parse_cli(arg)
    return isempty(arg) ? "20261106_amp_phs_corr_" : arg
end

# ========================================================
# Main
# ========================================================

function main()
    base_name = parse_cli(ARGS)

    df = CSV.read(SUMMARY_CSV, DataFrame)

    filtered = subset(df, :name => ByRow(n -> startswith(n, base_name)))
    isempty(filtered) && error("No rows found starting with $(base_name)")

    filtered[!, :infidelity] = 1.0 .- filtered[!, :avg_fidelity]
    sort!(filtered, [:n_qubits, :beta])

    n_qubits = sort(unique(filtered[!, :n_qubits]))
    @debug "Filtered rows" nrows(filtered) n_qubits

    # --- Summary -------------------------------------------
    println("\n=== SUMMARY [$(base_name)] " * "==="^10)
    println(rpad("n_qubits", 10), rpad("beta", 10), rpad("avg_fidelity", 16), "infidelity")
    for row in eachrow(sort(filtered, [:n_qubits, :beta]))
        println(
            rpad(row.n_qubits, 10),
            rpad(row.beta, 10),
            rpad(round(row.avg_fidelity; digits=6), 16),
            round(row.infidelity; digits=6),
        )
    end

    # --- Plot ----------------------------------------------
    gr()


    plt = plot(
        title="Average Infidelity",
        xlabel=L"\beta",
        ylable=L"1 - \bar{\mathcal{F}}",
        yscale=:log10,
        ylims=[0.01, 1.0],
        legend=:topright,
        framestyle=:box,
        grid=true,
        minorgrid=true,
        size=(750, 460),
        dpi=150,
        margin=4Plots.mm,
    )

    for (i, n) in enumerate(n_qubits)
        sub = subset(filtered, :n_qubits => ByRow(==(n)))
        scatter!(plt,
            sub[!, :beta],
            sub[!, :infidelity],
            label="n = $n",
            markershape=MARKER_SHAPES[mod1(i, length(MARKER_SHAPES))],
            markersize=7,
            markerstrokewidth=0.5,
            markerstrokecolor=:auto,
        )
    end

    # --- Save -------------------------------------------------
    out_path = joinpath(EXPERIMENTS_DIR, "$(base_name)infidelity")
    savefig(plt, "$(out_path).png")
    savefig(plt, "$(out_path).pdf")
    println("Saved -> $out_path")

    return filtered

end
