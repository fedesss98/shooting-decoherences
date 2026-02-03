"""
Create the collisional Unitary Evolution that gives back a recovery map on the system.
"""

include("../src/utils.jl")
using DecoKiller.UnitaryDilation
using DecoKiller.PetzMaps

using LinearAlgebra
using StatsBase
using ProgressBars

# System Variables
const d_S = 2  # System dimension (qubit)
const d_A = 2  # Ancilla dimension

const GAMMA     = 1.0
const TIMESTEPS = 11
const DT        = 0.1
const TIME      = 1.0
const N_STATES = 16 # How many states generated
const N_QUBITS = 1  # Qubits to create a logic qubit
const BETA = 0.5

noise_models = [
    (0.3, "bitflip"),
    (0.5, "dephasing"),
]

function get_kruas_operators(noise, gamma, t)
    if noise == "amplitude_damping"
        return get_amplitudedamping_operators(gamma, t)
    elseif noise == "dephasing"
        return get_dephasing_operators(gamma, t)
    elseif noise == "bitflip"
        return get_bitflip_operators(gamma, t)
    else
        error("Unknown noise model: $noise")
    end
    
end

function forward_evolution(model, ρ)
    kraus = model.kraus_fwd
    ρf = apply_channel(kraus, ρ, N_QUBITS)
    return ρf
end

function recovery(model, ρ)
    ρr, η = apply_petz_collision(model, ρ)
    return ρr, η
end

function main()
    # Create the reference state sigma
    sigma = thermal_state(N_QUBITS, BETA)
    fidelities = Float64[]
    # State to recover
    ρ0 = sigma  # Start from the reference state  
    ρi = ρ0      
    # Choose randomly a noise model
    noise = sample([n[2] for n in noise_models], Weights([n[1] for n in noise_models]))
    println("Selected noise model: $noise")

    time_iter = ProgressBar(1:DT:TIMESTEPS)
    for t in time_iter
        set_description(time_iter, "$(round(t, digits=2))s")
        # Get the forward Kraus operators for the randomly selected channel
        kraus_fwd = get_kruas_operators(noise, GAMMA, t)

        # Create the Petz Collision Model
        petz_model = PetzCollisionModel(kraus_fwd, sigma)

        # Evolve the system and recover it
        ρf = forward_evolution(petz_model, ρi)
        ρr, η = recovery(petz_model, ρf)

        # Compute Fidelity and print results
        fid_noise = fidelity(ρ0, ρf)
        fid_recovery = fidelity(ρ0, ρr)
        push!(fidelities, fid_noise)
        push!(fidelities, fid_recovery)

        ρi = ρr

    end
    return fidelities
end

main()
