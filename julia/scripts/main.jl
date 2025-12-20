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

using DecoKiller.PetzMaps # Custom function
using LinearAlgebra
using Plots       
using ProgressBars        # Show loop progresses
using LaTeXStrings        # Pretty print Plots labels
using Random              # Generate random states
using JSON                # Save outputs


const GAMMA    = 1.0
const TIMES    = range(0.0, 1, 101)  # The number of items should be even
const N_STATES = 16 # How many states generated
const N_QUBITS = 5 # Qubits to create a logic qubit
const BETA = 0.5

const I2 = [1.0+0.0im 0.0; 0.0 1.0]
# Ground and excited states of one qubit
const g0 = [0.0 + 0.0im; 1.0]
const e1 = [1.0 + 0.0im; 0.0]


function fullrank_state(n_qubits)
    Random.seed!(1234)
    coeffs = normalize(randn(Float64, 2^n_qubits))
    println(coeffs)
    r = zeros(ComplexF64, 2^n_qubits, 2^n_qubits)
    for (i, c) in enumerate(coeffs)
        psi = zeros(ComplexF64, 2^n_qubits)
        psi[i] = c
        r += psi * psi'
    end
    return r
end


function starting_state(n, beta)
    σz = [1.0+0.0im 0; 0 -1]
    q = zeros(ComplexF64, 2^n, 2^n)
    for i in 1:n
        for j in 1:i-1
            σz_i = foldl(kron, [k == i ? σz : I2 for k in 1:n])
            σz_j = foldl(kron, [k == j ? σz : I2 for k in 1:n])
            q += σz_i * σz_j
        end
    end
    return exp(beta * q) / tr(exp(beta * q))

end


function input_state(n, a, b)
    ground = foldl(kron, [g0 for _ in 1:n])
    excited = foldl(kron, [e1 for _ in 1:n])
    state = normalize(a * ground + b * excited)
    return state
end


function get_kraus_operators(process, gamma, t)
    if process == "dephasing"
        kraus = get_dephasing_operators(gamma, t)
    elseif process == "damping"
        kraus = get_amplitudedamping_operators(gamma, t)
    elseif process == "bitflip"
        kraus = get_bitflip_operators(gamma, t)
    end
    return kraus 
end


function random_state(n_qubits)
    # Sample uniformly in the sphere
    ϕ = 2 * rand() * π
    z = 2 * rand() - 1

    θ = acos(z)

    α = cos(θ / 2)
    β = exp(im * ϕ) * sin(θ / 2)

    return input_state(n_qubits, α, β)
end


function run_experiment(
    noise_model="bitflip"; 
    n_qubits  = N_QUBITS,
    n_states  = N_STATES,
    beta      = BETA,
    gamma     = GAMMA,
    times     = TIMES,
)

    if !in(noise_model, ["dephasing", "damping", "bitflip"])
        error("Process not known, try with `dephasing` or `decoherence`!")
    end

    println("==== STARTING experiment with parameters ====")
    for (key, val) in sort(collect(Base.@locals), by=x->x[1])
        # rpad aligns the keys to 15 characters for a clean column
        println("$(rpad(key, 15)) : $val")
    end
    println("============================")

    f1s = []
    f2s = []
    states = []
    avg_fs = Dict(t=>0.0 for t in times)

    γ = starting_state(n_qubits, beta)
    ψ0 = zeros(ComplexF64, 2^n_qubits)
    
    for i in ProgressBar(1:n_states)
        if i == 1
            ρ0 = deepcopy(γ)
            a=b=-1
        else
            ψ0 = random_state(n_qubits)
            ρ0 = ψ0 * ψ0'
        end
        ρ1 = deepcopy(ρ0)
        ρ2 = deepcopy(ρ0)

        f1s_sigma = []
        f2s_sigma = []
        for t in times

            # Evolve the states
            kraus = get_kraus_operators(noise_model, gamma, t)
            ρ1 = apply_channel(kraus, ρ1, n_qubits)
            ρ2 = apply_channel(kraus, ρ2, n_qubits)
            # Recover only the second state
            ρ2 = recovery_map(kraus, γ, ρ2, n_qubits)

            # Compute fidelity
            f1 = fidelity(ρ0, ρ1)
            f2 = fidelity(ρ0, ρ2)

            append!(f1s_sigma, f1)
            append!(f2s_sigma, f2)
            if i!=1
                avg_fs[t] += overlap(ρ2, ψ0) / n_states
            end
        end
        push!(f1s, f1s_sigma)
        push!(f2s, f2s_sigma)
        push!(states, ψ0)
    end
    avg_f_values = collect(values(avg_fs))[2:end]
    avg_f = sum(avg_f_values) / (length(avg_f_values))
    std_f = sum((x - avg_f)^2 for x in avg_f_values) / (length(avg_f_values) - 1)
    println("Average Fidelity: $avg_f +- $std_f")

    # Extract metadata to ensure consistent ordering
    titles = ["Fidelity Evolution ψ$i" for i in 1:n_states]
    
    n_plot_rows = 4
    n_plot_cols = 4
    n_plots = minimum([n_plot_cols * n_plot_rows, length(f1s)])
    plot_array = [
        plot(times, [f1s[i], f2s[i]],
             title = titles[i],
             label = ["Free" "Recovered"],
             legend = :topright)
        for i in 1:n_plots
    ]

    
    p = plot(plot_array...,
             layout=(n_plot_rows, n_plot_cols),
             size=(1200, 800),
             plot_title = "Recovery vs $(noise_model) noise (N=$n_qubits), β=$beta")

    println("\n=== END ===\n")
    #=savefig(p, "../visualization/$(noise_model)/$(n_qubits)qubits_recovery_in_time_beta$beta.pdf")=#
    savefig(p, "../visualization/$noise_model/$(n_qubits)qubits_recovery_in_time_beta$beta.png")

    
    return f1s, f2s, states, avg_fs
end


function plot_fig3(recovery_fidelities, k; logy=false)
    if !(k in keys(recovery_fidelities[1][3]))
        k = collect(keys(recovery_fidelities[1][3]))[k]
    end
    ns = unique(n for (n, _, _) in recovery_fidelities)
    xs = [[n*beta for (n, beta, _) in recovery_fidelities if n == qbts]
           for qbts in ns]
    ys = [[1 - f[k] for (n, _, f) in recovery_fidelities if n==qbts]
            for qbts in ns]

    p = scatter(xs, ys, 
                plot_title = "Average Infidelity vs β",
                xlabel = L"$\beta$", ylabel=L"$1-\mathcal{F}_k$",
                ylims=(0.0, 1.0),
                size=(600, 400))
    if logy
        # Define log scale values
        tick_values = 10.0 .^ (-10:1:0)

        # Define corresponding labels using LaTeX syntax
        tick_labels = ["10^{$i}" for i in -10:1:0]
        plot!(yscale=:log10, yticks=(tick_values, tick_labels))
    end

    display(p)
    #=savefig(p, "../visualization/nb_vs_k$k.pdf")=#
    savefig(p, "../visualization/infidelity/nb_vs_fk$k.png")

    println("Fig 3 saved.")

    return p
end


function iterate_exps(betas, n_qubits; noise_model="bitflip", save=true, kwargs...)
    f1b = []
    
    i = 0
    for n in n_qubits
    for b in betas
        i += 1
        println("Experiment $i")
        f1s, f2s, states, avg = run_experiment(noise_model; beta=b, n_qubits=n, kwargs...)
        push!(f1b, (n, b, avg))
    end
    end

    if save
        open("../experiments/nbeta_iterations/average_recovery_fidelities.json", "w") do f
            JSON.print(f, f1b)
        end
    end

    return f1b
end
