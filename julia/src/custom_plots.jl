using Plots.Measures
using Plots


function get_dual_intervals(mask::Vector{Int})
    # Returns vector of tuples (start, end) for both 0s and 1s
    # Uses Float64 for bounds to meet exactly in the middle (e.g., 2.5)

    intervals_a = Tuple{Float64,Float64}[]
    intervals_b = Tuple{Float64,Float64}[]

    if isempty(mask)
        return intervals_a, intervals_b
    end

    # Start the first boundary at 0.5 so the first point (1.0) is centered
    current_start = 1.5
    current_val = mask[1]

    for i in 2:length(mask)
        if mask[i] != current_val
            # A switch happened between i-1 and i
            # We set the boundary at i - 0.5
            boundary = i - 0.5

            if current_val == 1 # Previously was 1 (Choice A, for example)
                push!(intervals_a, (current_start, boundary))
            else                # Previously was 0 (Choice B)
                push!(intervals_b, (current_start, boundary))
            end

            # Start the new run from the boundary
            current_start = boundary
            current_val = mask[i]
        end
    end

    # Close the final interval
    # Extend to length + 0.5 to cover the last point
    final_end = length(mask) + 0.5
    if current_val == 1
        push!(intervals_a, (current_start, final_end))
    else
        push!(intervals_b, (current_start, final_end))
    end

    return intervals_a, intervals_b
end

function plot_autorecovery(state, cfg, logs; choices=true, show=true, save=false, kwargs...)
    ylims = get(kwargs, :ylims, [0.8, 1.01])
    xlims = get(kwargs, :xlims, [0, cfg.n_timesteps])
    size = get(kwargs, :size, (1200, 400))

    real_noise_label = titlecase(join(split(cfg.real_noise.name, '_'), ' '))
    noise_labels = [
        titlecase(join(split(state.noise_options[1].name, '_'), ' ')),
        titlecase(join(split(state.noise_options[2].name, '_'), ' '))
    ]

    p = plot(
        ylim=ylims, xlim=xlims, size=size, legend=:outertopright,
        titlefontsize=18,
        guidefontsize=18,
        tickfontsize=16,
        legendfontsize=12,
        grid=false,
        left_margin=10mm,
        top_margin=10mm,
        bottom_margin=9mm
    )

    if choices
        label_a = 0
        label_b = 0
        vline!(p, collect(0:length(logs.ref_fidelities)) .+ 1.2, linestyle=:dash, color=:lightgrey, lw=1, label=nothing)

        for i in 1:(length(state.choice.history)-1)
            current_choice = state.choice.history[i]
            band_color = state.choice.history[i] == 1 ? :lightblue : :lightgreen
            if label_a == 0 && current_choice == 1
                label = "$(noise_labels[1]) Recovery"
                label_a += 1
            elseif label_b == 0 && current_choice == 2
                label = "$(noise_labels[2]) Recovery"
                label_b += 1
            else
                label = nothing
            end
            vspan!(p, [i + 0.2, i + 1.2], color=band_color, alpha=0.2, label=label)
        end
    end
    
    scatter!(
        [[1.0; logs.ref_fidelities] [1.0; logs.fidelities]],
        labels=["Free evolution" "Recovered Evolution"],
        xlabel="Timestep", ylabel="Fidelity",
        shape=[:dtriangle :circle],
        markercolor=[:steelblue :red],
        markerstrokecolor=[:steelblue :transparent],
        markerstrokewidth=2, markersize=[7 5],
        title="Adaptive Recovery under $(real_noise_label) Noise with $(cfg.n_qubits) Qubits"
    )

    if save
        output_folder = joinpath(cfg.experiment_dir, "visualization")
        savefig(p, joinpath(output_folder, "adaptive_recovery.png"))
        savefig(p, joinpath(output_folder, "adaptive_recovery.pdf"))
    end

    if !show
        return p
    end

    display(p)
end

function plot_average_fidelity(avg_fidelities, avg_ref_fidelities, state, cfg; show=true, save=false, kwargs...)
    ylims = get(kwargs, :ylims, [0.0, 1.01])
    xlims = get(kwargs, :xlims, [0, cfg.n_timesteps])
    size = get(kwargs, :size, (1200, 400))
    infidelity = get(kwargs, :infidelity, false)

    real_noise_label = titlecase(join(split(cfg.real_noise.name, '_'), ' '))

    p = plot(
        ylim=ylims, xlim=xlims, size=size, 
        legend=get(kwargs, :legend, :outertopright),
        titlefontsize=18,
        guidefontsize=18,
        tickfontsize=16,
        legendfontsize=12,
        grid=false,
        left_margin=10mm,
        top_margin=10mm,
        bottom_margin=9mm,
        yscale=infidelity ? :log10 : :identity
    )

    vline!(p, collect(0:length(avg_ref_fidelities)) .+ 1.2, linestyle=:dash, color=:lightgrey, lw=1, label=nothing)

    if infidelity
        avg_ref_fidelities = 1 .- avg_ref_fidelities
        avg_fidelities = 1 .- avg_fidelities
    end

    scatter!(
        [[1.0; avg_ref_fidelities] [1.0; avg_fidelities]],
        labels=["Free evolution" "Recovered Evolution"],
        xlabel="Timestep",
        ylabel=infidelity ? "Infidelity" : "Fidelity",
        shape=[:dtriangle :circle],
        markercolor=[:steelblue :red],
        markerstrokecolor=[:steelblue :transparent],
        markerstrokewidth=2, markersize=[7 5],
        title="Adaptive Recovery under $(real_noise_label) Noise with $(cfg.n_qubits) Qubits"
    )

    if save
        output_folder = joinpath(cfg.experiment_dir, "visualization")
        plot_title = get(kwargs, :plot_title,
            infidelity ? "adaptive_recovery_infidelity_avg" : "adaptive_recovery_avg"
        )
        @debug "Saving plot" output_folder plot_title
        savefig(p, joinpath(output_folder, "$(plot_title).png"))
        savefig(p, joinpath(output_folder, "$(plot_title).pdf"))
    end

    if !show
        return p
    end

    display(p)

end