"""
Create the collisional Unitary Evolution that gives back a recovery map on the system.
"""

include("../src/utils.jl")
using DecoKiller.UnitaryDilation
using DecoKiller.PetzMaps

using LinearAlgebra
using StatsBase
using ProgressBars
using Dates

# System Variables
const d_S = 2  # System dimension (qubit)
const d_A = 2  # Ancilla dimension

const GAMMA = 1.0
const TIMESTEPS = 3
const DT = 0.1
const TIME = 1.0
const N_STATES = 16 # How many states generated
const N_QUBITS = 1  # Qubits to create a logic qubit
const BETA = 0.5

noise_models = [
  (0.3, "bitflip"),
  (0.5, "dephasing"),
]

function get_kraus_operators(noise, gamma, t)
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


function kraus_to_superop(kraus_ops)
    d = size(kraus_ops[1], 1)
    superop = zeros(ComplexF64, d^2, d^2)
    for K in kraus_ops
        superop += kron(conj(K), K)
    end
    return superop
end

"""
  build_step_matrix(model)
Builds the superoperator matrix which implements the n+1 evolution step:
collision + noise
"""
function build_step_matrix(model)
  M_petz = kraus_to_superop(model.kraus_rec)
  M_noise = kraus_to_superop(model.kraus_fwd)

  return M_petz, M_noise
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
  current_kraus = get_kraus_operators(noise, GAMMA, DT)
  current_channel = ρ -> apply_channel(current_kraus, ρ, N_QUBITS)

  time_iter = ProgressBar(1:DT:TIMESTEPS)
  for t in time_iter
    set_description(time_iter, "$(round(t, digits=2))s")

    # Get the forward Kraus operators for the inferred noise
    step_noise = ρ -> apply_channel(current_kraus, ρ, N_QUBITS)
    current_channel = add_petz_noise_layer(current_channel, step_noise, sigma, 2^N_QUBITS)
    kraus = extract_kraus_operators(current_channel, 2^N_QUBITS)
    # Create the Petz Collision Model
    petz_model = PetzCollisionModel(kraus, sigma)

    # Evolve the system and recover it
    ρf = current_channel(ρi)
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

# main()

function prove_one_recovery()
  println(now())
  # Create the reference state sigma
  sigma = thermal_state(N_QUBITS, BETA)
  println("Reference state σ:")
  display(sigma)
  psi = random_state(N_QUBITS)
  ρ0 = psi * psi'  # Start from a random pure state
  println("Initial random state ρ0:")
  display(ρ0)
  
  # Choose randomly a noise model
  kraus_fw = get_kraus_operators("amplitude_damping", GAMMA, 10*DT)

  # Step 1
  # N = Omega_a
  println("\n-- STEP 1: Noise channel = Ω[ρ0]")
  
  petz_model = PetzCollisionModel(kraus_fw, ρ0)
  ρf = forward_evolution(petz_model, ρ0)
  ρr, _ = recovery(petz_model, ρf)
  # Compute Fidelity
  println("Fidelity after noise: $(fidelity(ρ0, ρf))")
  println("Fidelity after recovery: $(fidelity(ρ0, ρr))")
  
  # Step 2 
  # N = Omega_a ∘ PetzCollision ∘ Omega_a
  println("\n-- STEP 2: Noise channel = Ω[Tr[U Ω[ρ0]⊗|0><0| U†]]")

  M_petz, M_noise = build_step_matrix(petz_model)
  M_total = M_noise * M_petz * M_noise
  petz_model = PetzCollisionModel(M_total, ρ0)

  ρf = forward_evolution(petz_model, ρr)
  ρr, _ = recovery(petz_model, ρf)
  # Compute Fidelity
  println("Fidelity after noise (2 steps): $(fidelity(ρ0, ρf))")
  println("Fidelity after recovery (2 steps): $(fidelity(ρ0, ρr))")

end


function prove_autorecovery(; n=2)
  println(now())
  println("==== AUTORECOVERY TEST ====")
  println("We test that a reference state σ is recovered")
  println(" after the action of noise, using a collisional")
  println(" Petz map aided by an ancilla.")
  println("\nWe do this iteratively for 2 steps, considering")
  println(" the action of the collision in the second step.")

  # Create the reference state sigma
  # WARNING: only some state are well recovered, depending on the noise
  sigma = thermal_state(N_QUBITS, BETA)
  ρ0 = sigma
  # Choose randomly a noise model
  kraus_fw = get_kraus_operators("amplitude_damping", GAMMA, 10*DT)

  # Step 1
  # N = Omega_a
  println("\n-- STEP 1: Noise channel = Ω[σ]")
  
  petz_model = PetzCollisionModel(kraus_fw, ρ0)
  ρf = forward_evolution(petz_model, ρ0)
  ρr, _ = recovery(petz_model, ρf)
  # Compute Fidelity
  println("Fidelity after noise: $(fidelity(sigma, ρf))")
  println("Fidelity after recovery: $(fidelity(sigma, ρr))")
  
  # Step 2 
  # N = Omega_a ∘ PetzCollision ∘ Omega_a
  println("\n-- STEP 2: Noise channel = Ω[Tr[U Ω[σ]⊗|0><0| U†]]")

  M_petz, M_noise = build_step_matrix(petz_model)
  M_total = M_noise * M_petz * M_noise
  petz_model = PetzCollisionModel(M_total, ρ0)

  ρf = forward_evolution(petz_model, ρr)
  ρr, _ = recovery(petz_model, ρf)
  # Compute Fidelity
  println("Fidelity after noise (2 steps): $(fidelity(sigma, ρf))")
  println("Fidelity after recovery (2 steps): $(fidelity(sigma, ρr))")

end