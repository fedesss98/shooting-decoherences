"""
Study the Petz Recovery Map (PRM) Λ on the Amplitude Damping Channel (ADC).
- select one initial state σ;
- find the PRM to invert the action of the ADC after some time τ;
- select a second state ρ undergoing the same ADC effect;
- use the PRM on ρ(τ);
- compute the Fidelities F1(ρ, ρ(τ)) and F2(ρ, Λ[ρ(τ)]);
- see if there is an improvement in keeping the state ρ intact using the PRM;
- iterate for different initial state σ and different probe state ρ.
"""

using DecoKiller.PetzMaps
using LinearAlgebra
using Plots
using ProgressBars

gamma = 1.0
times = range(0.001, 10, 101)  # The number of items should be even
n_states = 16 # How many states generated
n_qubits = 3 # Qubits to create a logic qubit


function fullrank_state(n_qubits)
    coeffs = normalize(randn(Float64, 2^n_qubits))
    r = zeros(ComplexF64, 2^n_qubits, 2^n_qubits)
    for (i, c) in enumerate(coeffs)
        psi = zeros(ComplexF64, 2^n_qubits)
        psi[i] = c
        r += psi * psi'
    end
    return r
end


function get_kraus_operators(process, gamma, t)
    if process == "dephasing"
        kraus = get_dephasing_operators(gamma, t)
    elseif process == "damping"
        kraus = get_amplitudedamping_operators(gamma, t)
    end
    return kraus 
end


function main(process="dephasing")
    if !in(process, ["dephasing", "damping"])
        error("Process not known, try with `dephasing` or `decoherence`!")
    end

    println("==== STARTING ====")

    f1s = Dict()
    f2s = Dict()
    σ = fullrank_state(n_qubits)
    for i in ProgressBar(1:n_states)
        a, b, c, d = normalize(randn(Float64, 4))
        if i==1
            # Start with the reference state itself
            a = 1.0
            b = 0.0
            c = 0.0
            d = 0.0
        end
        ρ0 = codespace_state(n_qubits, a, b, c, d)
        ρ1 = deepcopy(ρ0)
        ρ2 = deepcopy(ρ0)


        f1s_sigma = []
        f2s_sigma = []
        for t in times

            # Compute fidelity
            f1 = fidelity(ρ0, ρ1)
            f2 = fidelity(ρ0, ρ2)
            # Evolve the state
            kraus = get_kraus_operators(process, gamma, t)
            ρ1 = apply_channel(kraus, ρ1, n_qubits)
            # Recover
            ρ2 = deepcopy(ρ1)
            ρ2 = recovery_map(kraus, σ, ρ2, n_qubits)

            append!(f1s_sigma, f1)
            append!(f2s_sigma, f2)
        end

        f1s[(a,b)] = f1s_sigma
        f2s[(a,b)] = f2s_sigma
    end

    # Extract metadata to ensure consistent ordering
    titles = Dict((a,b) => "$(round(a, digits=3)) / $(round(b, digits=3))" for (a, b) in keys(f1s))

    plot_array = [
        plot(times, [f1s[k], f2s[k]],
             title = titles[k],
             label = ["Free" "Recovered"],
             legend = :topright)
        for k in keys(f1s)
    ]

    p = plot(plot_array...,
             layout=(4, 4),
             size=(1200, 800),
             plot_title = "Recovery vs Dephasing with fullrank state")

    display(p)

    println("=== END ===")
    savefig(p, "../visualization/$(n_qubits)qubits_recovery_in_time.pdf")
    savefig(p, "../visualization/$(n_qubits)qubits_recovery_in_time.png")

    
    return f1s, f2s
end


function plot_heatmap(f1s, f2s; title="", save=false)
    a1_vals = range(-1, 1, n_states)
    a2_vals = range(-1, 1, n_states)
    # Convert dict to matrix
    f1_matrix = hcat([f1s[a1] for a1 in a1_vals]...)'
    f2_matrix = hcat([f2s[a1] for a1 in a1_vals]...)'
    metric = f1_matrix - f2_matrix
    # Heatmap
    #=heatmap(a1_vals, a2_vals, metric, =#
    #=        xlabel="a1", ylabel="a2", =#
    #=        title="f(a1, a2)")=#

    # Or contour plot
    g = contour(a1_vals, a2_vals, metric,
            xlabel="a₁", ylabel="a₂",
            title=title,
            fill=true)    

    if save
        savefig(g, "../visualization/recovery_contour.pdf")
        savefig(g, "../visualization/recovery_contour.png")

        display(g)
    end
    return g
end


function heatmap_subplots(temporal_evolutions)
    plots_array = []

    for (t, (f1s, f2s)) in temporal_evolutions
        p = plot_heatmap(f1s, f2s; title="τ = $t")
        push!(plots_array, p)
    end

    n = length(plots_array)
    rows = ceil(Int, n / 2)
    g = plot(plots_array..., layout=(rows, 2), size=(1600, 1200), link=:both)
    savefig(g, "../visualization/recovery_contour_grid.pdf")
    savefig(g, "../visualization/recovery_contour_grid.png")
    display(g)
end
