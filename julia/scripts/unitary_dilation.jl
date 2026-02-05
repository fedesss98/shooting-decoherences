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
using Random

# System Variables
const d_S = 2  # System dimension (qubit)
const d_A = 2  # Ancilla dimension

const GAMMA = 1.0
const TIMESTEPS = 10
const DT = 0.1
const TIME = 1.0
const N_STATES = 16 # How many states generated
const N_QUBITS = 1  # Qubits to create a logic qubit
const BETA = 0.5

noise_models = [
  (0.75, "amplitude_damping"),
  (0.25, "dephasing"),
]

struct NoiseObj{T<:Number}
  name::String
  probability::Float64
  kraus::Vector{Matrix{T}}
  supermap_petz::Matrix{T}
  supermap_noise::Matrix{T}
end

function NoiseObj(noise::String, p::Float64, gamma::Float64, sigma::Matrix{T}, t::Float64) where T
  kraus = get_kraus_operators(noise, gamma, t)
  model = PetzCollisionModel(kraus, sigma)
  M_petz, M_noise = build_superoperators(model)
  return NoiseObj{T}(noise, p, kraus, M_petz, M_noise)
end

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


function discrimin(ρ_test, ρ_options...)
  v = 0.25 / (length(ρ_options) - 1)
  return [0.75, fill(v, length(ρ_options) - 1)...]
end


function main(;
  n_qubits      = N_QUBITS,
  noise_models  = noise_models,
  gamma         = GAMMA,
  beta          = BETA,
  timesteps     = TIMESTEPS,
  dt            = DT,
  seed          = nothing
)
  rng = isnothing(seed) ? Random.default_rng() : Xoshiro(seed)  # For reproducibility
  fidelities = Float64[]
  noise_options = NoiseObj{ComplexF64}[]
  c1 = c2 = 0


  # Choose randomly a noise model
  noise = sample(rng,
    [n[2] for n in noise_models], 
    Weights([n[1] for n in noise_models])
  )
  println("\nSELECTED NOISE MODEL: $noise")
  
  # Create the reference state sigma
  sigma = thermal_state(n_qubits, beta)
  println("\nReference state σ:")
  display(sigma)
  # Start from a random pure state
  psi = random_state(n_qubits; seed=seed)
  ρ0 = psi * psi'  
  println("\nInitial random state ρ0:")
  display(ρ0)

  # Precompute the supermaps
  noise_options = [
    NoiseObj(noise_model[2], noise_model[1], gamma, sigma, dt)
    for noise_model in noise_models
  ]

  # T1 : initialization

  # 1. REAL Noise is applied to the state
  real_kraus = get_kraus_operators(noise, gamma, dt)
  real_model = PetzCollisionModel(real_kraus, sigma)
  ρf = apply_noise(real_model, ρ0, n_qubits)
  # 2. Take an initial guess for the noise model
  noise_guess = sample(rng, [n[2] for n in noise_models])
  println("\nInitial noise guess: $noise_guess\n")
  kraus_fwd = get_kraus_operators(noise_guess, gamma, dt)
  # 3. Build the Petz Collision Model
  petz_model = PetzCollisionModel(kraus_fwd, sigma)
  # 4. Act: recover the state
  ρr, η = recovery(petz_model, ρf)
  # 5. Compare with possible ancilla outputs
  ancilla_options = []
  for option in noise_options
    _model = PetzCollisionModel(option.kraus, sigma)
    _, _η = recovery(_model, ρf)
    push!(ancilla_options, _η)
  end
  # 6. Discriminate the ancilla states
  weights = discrimin(η, ancilla_options...)
  println("Discrimination weights: $weights")
  count = sample(rng, [1, 2], Weights(weights))
  # update the count of noise choices
  if count == 1
    c1 += 1
  else
    c2 += 1
  end

  # 7. Choose the next noise model accordingly
  if c1 > c2 || 10*(c1 - c2) == (count - 2)
    # 7a : assume the first noise model
    noise_guess = noise_options[1]
  elseif c2 > c1 || 10*(c1 - c2) == (count - 1)
    # 7b : assume the second noise model
    noise_guess = noise_options[2]
  end
  # 8. Construct the complete map up to this point
  M_petz, M_noise = noise_guess.supermap_petz, noise_guess.supermap_noise
  M_total = (M_noise * M_petz) * M_noise
  # 9. Update the Petz Collision Model
  petz_model = PetzCollisionModel(M_total, sigma)
  
  # 10. Compute Fidelity and print results
  fid_noise = fidelity(ρ0, ρf)
  fid_recovery = fidelity(ρ0, ρr)
  push!(fidelities, fid_noise)
  push!(fidelities, fid_recovery)
  println("Fidelity after noise: $fid_noise")
  println("Fidelity after recovery: $fid_recovery")

  # 11. Iterate the process
  final_t = 1 + timesteps * dt
  time_iter = ProgressBar(1:dt:final_t)
  step = 1
  for t in time_iter
    set_description(time_iter, "$(round(t, digits=2))s")

    # 1. REAL Noise is applied to the state
    ρf = apply_noise(real_model, ρr, n_qubits)
    # 2. Act: recover the state
    ρr, η = recovery(petz_model, ρf)
    # 3. Compare with possible ancilla outputs
    ancilla_options = []
    for option in noise_options
      _model = PetzCollisionModel(option.kraus, sigma)
      _, _η = recovery(_model, ρf)
      push!(ancilla_options, _η)
    end
    # 4. Discriminate the ancilla states
    weights = discrimin(η, ancilla_options...)
    count = sample(rng, [1, 2], Weights(weights))
    println(time_iter, "Measurement result: $count")
    # update the count of noise choices
    if count == 1
      c1 += 1
    else
      c2 += 1
    end

    # 5. Choose the next noise model accordingly
    if c1 > c2 || 10*(c1 - c2) == (count - 2)
      # 5a : assume the first noise model
      noise_guess = noise_options[1]
    elseif c2 > c1 || 10*(c1 - c2) == (count - 1)
      # 5b : assume the second noise model
      noise_guess = noise_options[2]
    end
    # 6. Construct the complete map up to this point
    M_petz, M_noise = noise_guess.supermap_petz, noise_guess.supermap_noise
    M_total = ((M_noise * M_petz)^step) * M_noise
    # 7. Update the Petz Collision Model
    petz_model = PetzCollisionModel(M_total, sigma)

    # 8. Compute Fidelity and print results
    fid_noise = fidelity(ρ0, ρf)
    fid_recovery = fidelity(ρ0, ρr)
    push!(fidelities, fid_noise)
    push!(fidelities, fid_recovery)
    println(time_iter, "Fidelity after noise: $fid_noise")
    println(time_iter, "Fidelity after recovery: $fid_recovery")

    step += 1
  end
  
  return fidelities
end

# main()

function prove_recovery(sigma=nothing;
  n=2, 
  noise="amplitude_damping", 
  seed=nothing,
  dt=DT,
  gamma=GAMMA,
  beta=BETA,
  n_qubits=N_QUBITS,
  )
  println(now())

  states = Matrix{ComplexF64}[]
  fidelities = Float64[]

  # Create the reference state sigma
  sigma = isnothing(sigma) ? thermal_state(n_qubits, beta) : sigma
  # sigma = random_state(n_qubits, seed=42)
  # sigma = sigma * sigma' 
  println("\nReference state σ:")
  display(sigma)
  psi = random_state(n_qubits, seed=seed)
  ρ0 = psi * psi'  # Start from a random pure state
  println("\nInitial random state ρ0:")
  display(ρ0)

  ρ1 = deepcopy(ρ0)
  ρ2 = deepcopy(ρ0)
  
  # Choose the noise model
  kraus_fwd = get_kraus_operators(noise, gamma, dt)

  # Step 1
  # N = Omega_a
  println("\n-- STEP 1: Noise channel = Ω[σ]")
  
  petz_model = PetzCollisionModel(kraus_fwd, sigma)

  ρ1 = apply_channel(petz_model.kraus_fwd, ρ1, n_qubits)
  ρ2 = apply_channel(petz_model.kraus_fwd, ρ2, n_qubits)
  # Enforce physicality (hermitianicity and trace 1)
  # enforce_physical!(ρf)
  ρ2, _ = apply_petz_collision(petz_model, ρ2)
  # enforce_physical!(ρr)
  # Compute Fidelity
  println("Fidelity after noise: $(fidelity(ρ0, ρ1))")
  println("Fidelity after recovery: $(fidelity(ρ0, ρ2))")
  
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

    ρ1 = apply_channel(petz_model.kraus_fwd, ρ1, n_qubits)
    ρ2 = apply_channel(petz_model.kraus_fwd, ρ2, n_qubits)
    # enforce_physical!(ρf)
    ρ2, _ = apply_petz_collision(petz_model, ρ2)
    # enforce_physical!(ρr)

    # Compute Fidelity
    println("Fidelity after noise: $(fidelity(ρ0, ρ1))")
    println("Fidelity after recovery: $(fidelity(ρ0, ρ2))")

    push!(fidelities, fidelity(ρ0, ρ1))
    push!(fidelities, fidelity(ρ0, ρ2))
    push!(states, ρ1)
    push!(states, ρ2)

  end

  # Display the final action of the channels
  println("\nFinal noisy state:")
  display(ρ1)
  println("\nFinal recovered state ρr:")
  display(ρ2)
  println("\n============================\n")

  return states, fidelities

end


function prove_classic_recovery(sigma=nothing;
  n=2, 
  noise="amplitude_damping", 
  seed=nothing,
  dt=DT,
  gamma=GAMMA,
  beta=BETA,
  n_qubits=N_QUBITS,
  )
  println(now())

  states = Matrix{ComplexF64}[]
  fidelities = Float64[]

  # Create the reference state sigma
  sigma = isnothing(sigma) ? thermal_state(n_qubits, beta) : sigma
  # sigma = random_state(n_qubits, seed=42)
  # sigma = sigma * sigma' 
  println("\nReference state σ:")
  display(sigma)
  psi = random_state(n_qubits, seed=seed)
  ρ0 = psi * psi'  # Start from a random pure state
  println("\nInitial random state ρ0:")
  display(ρ0)

  ρ1 = deepcopy(ρ0)
  ρ2 = deepcopy(ρ0)
  
  # Choose the noise model
  kraus = get_kraus_operators(noise, gamma, dt)

  # Step 1
  # N = Omega_a
  println("\n-- STEP 1: Noise channel = Ω[σ]")
  
  petz_model = PetzCollisionModel(kraus, sigma)

  ρ1 = apply_channel(kraus, ρ1, n_qubits)
  ρ2 = apply_channel(kraus, ρ2, n_qubits)
  # Recover only the second state
  ρ2 = recovery_map(kraus, sigma, ρ2, n_qubits)
  # Compute Fidelity
  println("Fidelity after noise: $(fidelity(ρ0, ρ1))")
  println("Fidelity after recovery: $(fidelity(ρ0, ρ2))")
  
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
    kraus = petz_model.kraus_fwd

    ρ1 = apply_channel(kraus, ρ1, n_qubits)
    ρ2 = apply_channel(kraus, ρ2, n_qubits)
    # Recover only the second state
    ρ2 = recovery_map(kraus, sigma, ρ2, n_qubits)

    # Compute Fidelity
    println("Fidelity after noise: $(fidelity(ρ0, ρ1))")
    println("Fidelity after recovery: $(fidelity(ρ0, ρ2))")

    push!(fidelities, fidelity(ρ0, ρ1))
    push!(fidelities, fidelity(ρ0, ρ2))
    push!(states, ρ1)
    push!(states, ρ2)

  end

  # Display the final action of the channels
  println("\nFinal noisy state:")
  display(ρ1)
  println("\nFinal recovered state ρr:")
  display(ρ2)
  println("\n============================\n")

  return states, fidelities

end


function prove_autorecovery(; 
  n=2, 
  noise="bitflip",
  dt = DT,
  gamma = GAMMA,
  beta = BETA,
  n_qubits = N_QUBITS,
  )
  println(now())
  println("==== AUTORECOVERY TEST ====")
  println("We test that a reference state σ is recovered after")
  println(" the action of noise, using a collisional Petz map and an ancilla.")
  println(" ")
  println("\nWe do this iteratively for 2 or more steps, considering")
  println(" the action of the collision in the second step.")

  # Create the reference state sigma
  # WARNING: only some state are well recovered, depending on the noise
  sigma = thermal_state(n_qubits, beta)
  ρ0 = sigma
  # Choose the noise model
  kraus_single_qubit = get_kraus_operators(noise, gamma, dt)

  # Step 1
  # N = Omega_a
  println("\n-- STEP 1: Noise channel = Ω[σ]")
  
  petz_model = PetzCollisionModel(kraus_single_qubit, ρ0, n=n_qubits)

  ρf = apply_channel(petz_model.kraus_fwd, ρ0)
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

    ρf = apply_channel(petz_model.kraus_fwd, ρr)
    enforce_physical!(ρf)
    ρr, _ = apply_petz_collision(petz_model, ρf)
    enforce_physical!(ρr)
    # Compute Fidelity
    println("Fidelity after noise: $(fidelity(ρ0, ρf))")
    println("Fidelity after recovery: $(fidelity(ρ0, ρr))")
  end
  println("\n============================\n")

end