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

function apply_noise(model, ρ, n_qubits)
  ρf = apply_channel(model.kraus_fwd, ρ, n_qubits)
  # Enforce physicality (hermitianicity and trace 1)
  enforce_physical!(ρf)
  return ρf
end


function recovery(model, ρ)
  ρr, η = apply_petz_collision(model, ρ)
  enforce_physical!(ρr)
  return ρr, η
end


function discrimin(ρ_test, ρ1, ρ2)
  return (0.75, 0.25)
end


function main(
  n_qubits      = N_QUBITS,
  noise_models  = noise_models,
  gamma         = GAMMA,
  beta          = BETA,
  timesteps     = TIMESTEPS,
  dt            = DT
)
  fidelities = Float64[]

  # Choose randomly a noise model
  noise = sample([n[2] for n in noise_models], Weights([n[1] for n in noise_models]))
  println("Selected noise model: $noise")

  # Create the reference state sigma
  sigma = thermal_state(N_QUBITS, BETA)
  println("\nReference state σ:")
  display(sigma)
  # Start from a random pure state
  psi = random_state(N_QUBITS)
  ρ0 = psi * psi'  
  println("\nInitial random state ρ0:")
  display(ρ0)

  # T1 : initialization

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

function prove_recovery(; n=2, noise="amplitude_damping")
  println(now())

  states = Matrix{ComplexF64}[]
  fidelities = Float64[]

  # Create the reference state sigma
  sigma = thermal_state(N_QUBITS, BETA)
  println("\nReference state σ:")
  display(sigma)
  psi = random_state(N_QUBITS)
  ρ0 = psi * psi'  # Start from a random pure state
  println("\nInitial random state ρ0:")
  display(ρ0)
  
  # Choose the noise model
  kraus_fwd = get_kraus_operators(noise, GAMMA, 10*DT)

  # Step 1
  # N = Omega_a
  println("\n-- STEP 1: Noise channel = Ω[σ]")
  
  petz_model = PetzCollisionModel(kraus_fwd, sigma)

  ρf = apply_channel(petz_model.kraus_fwd, ρ0, N_QUBITS)
  # Enforce physicality (hermitianicity and trace 1)
  enforce_physical!(ρf)
  ρr, _ = apply_petz_collision(petz_model, ρf)
  enforce_physical!(ρr)
  # Compute Fidelity
  println("Fidelity after noise: $(fidelity(ρ0, ρf))")
  println("Fidelity after recovery: $(fidelity(ρ0, ρr))")
  
  # Setup next evolutions
  _, M_total = build_superoperators(petz_model)

  for i in 2:n
    # N = Omega_a ∘ PetzCollision ∘ N
    println("\n-- STEP $i: Noise channel = Ω[Tr[U N[σ]⊗|0><0| U†]]")
    M_petz, M_noise = build_superoperators(petz_model)
    # Add the Petz collision and the noise to the total supermap
    M_total = M_noise * M_petz * M_total
    # Update the collision model
    petz_model = PetzCollisionModel(M_total, sigma)

    ρf = apply_channel(petz_model.kraus_fwd, ρr, N_QUBITS)
    enforce_physical!(ρf)
    ρr, _ = apply_petz_collision(petz_model, ρf)
    enforce_physical!(ρr)

    # Compute Fidelity
    println("Fidelity after noise: $(fidelity(ρ0, ρf))")
    println("Fidelity after recovery: $(fidelity(ρ0, ρr))")

    push!(fidelities, fidelity(ρ0, ρf))
    push!(fidelities, fidelity(ρ0, ρr))
    push!(states, ρf)
    push!(states, ρr)

  end

  println("\nFinal noisy state:")
  display(ρf)
  println("\nFinal recovered state ρr:")
  display(ρr)

  return states, fidelities

end


function prove_autorecovery(; n=2, noise="bitflip")
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
  # Choose the noise model
  kraus_fwd = get_kraus_operators(noise, GAMMA, 10*DT)

  # Step 1
  # N = Omega_a
  println("\n-- STEP 1: Noise channel = Ω[σ]")
  
  petz_model = PetzCollisionModel(kraus_fwd, ρ0)

  ρf = apply_channel(petz_model.kraus_fwd, ρ0, N_QUBITS)
  # Enforce physicality (hermitianicity and trace 1)
  enforce_physical!(ρf)
  ρr, _ = apply_petz_collision(petz_model, ρf)
  enforce_physical!(ρr)
  # Compute Fidelity
  println("Fidelity after noise: $(fidelity(ρ0, ρf))")
  println("Fidelity after recovery: $(fidelity(ρ0, ρr))")
  
  # Setup next evolutions
  _, M_total = build_superoperators(petz_model)

  for i in 2:n
    # N = Omega_a ∘ PetzCollision ∘ N
    println("\n-- STEP $i: Noise channel = Ω[Tr[U N[σ]⊗|0><0| U†]]")
    M_petz, M_noise = build_superoperators(petz_model)
    # Add the Petz collision and the noise to the total supermap
    M_total = M_noise * M_petz * M_total
    # Update the collision model
    petz_model = PetzCollisionModel(M_total, ρ0)

    ρf = apply_channel(petz_model.kraus_fwd, ρr, N_QUBITS)
    enforce_physical!(ρf)
    ρr, _ = apply_petz_collision(petz_model, ρf)
    enforce_physical!(ρr)
    # Compute Fidelity
    println("Fidelity after noise: $(fidelity(ρ0, ρf))")
    println("Fidelity after recovery: $(fidelity(ρ0, ρr))")
  end
end