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

using PetzCollisions
using LinearAlgebra
using Plots
using ProgressBars

gamma = 1.0
tau = 1.0
n_states = 10001 # How many states generated

kraus = get_kraus_operators(gamma, tau)

function main()
    println("==== STARTING ====")

    f1s = Dict()
    f2s = Dict()
    for a in ProgressBar(range(-1, 1, n_states))
        σ = density_matrix(a)
        f1s_sigma = []
        f2s_sigma = []
        for b in ProgressBar(range(-1, 1, n_states))
            ρ = density_matrix(b)
            
            ρt = amplitude_damping_channel(kraus, ρ)
            ρr = recovery_map(kraus, σ, ρt)

            f1 = fidelity(ρ, ρt)
            f2 = fidelity(ρ, ρr)

            append!(f1s_sigma, f1)
            append!(f2s_sigma, f2)
        end
        f1s[a] = f1s_sigma
        f2s[a] = f2s_sigma
    end

    x = range(-1, 1, n_states)
    a = 0.0 
    g = plot(x, [f1s[a], f2s[a]], label=["F1(ρ(0), ρ(τ))" "F2(ρ(0), Λ[ρ(τ)])"],
             xlabel="State parameterization (a|0> + √(1-a²)|1>",
             ylabel="Fidelity", title="PRM created with a=$a")
    println("=== END ===")
    display(g)
    savefig(g, "visualization/recovery_from_sigma_a1_$a.pdf")
    savefig(g, "visualization/recovery_from_sigma_a1_$a.png")

    return f1s, f2s
end

function plot_heatmap(f1s, f2s)
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
            xlabel="a₁ (σ(0) from a₁|0> + √(1-a₁²)|1>)", ylabel="a₂ (ρ(0) from a₂|0> + √(1-a₂²)|1>)",
            title="F1(ρ(0), ρ(τ)) - F2(ρ(0), Λ[ρ(τ)])",
            fill=true)    


    savefig(g, "visualization/recovery_contour.pdf")
    savefig(g, "visualization/recovery_contour.png")

    display(g)
end
