"""
    discrim(ρ_test, ρ₁, ρ₂)

Performs optimal discrimination between two reference quantum states ρ₁ and ρ₂
given a test state ρ_test. Returns the probability that ρ_test is state i.

# Arguments
- `ρ_test`: The test density matrix to discriminate
- `ρ₁`: First reference density matrix
- `ρ₂`: First reference density matrix

# Returns
- `(p₁, p₂)`: Probabilities that the test state is ρ₁ or ρ₂ respectively

# Algorithm
Uses the Helstrom measurement (optimal POVM for minimum error discrimination):
- Π₁ = projector onto positive eigenspace of (ρ₁ - ρ₂)
- Π₂ = I - Π₁
- Returns pᵢ = Tr(Πᵢ * ρ_test)
"""
function discrimin(ρ_test, ρ1, ρ2; tol=1e-10)
    d = size(ρ_test, 1)
    Δρ = ρ1 - ρ2

    # Early exit: states are indistinguishable, return uniform
    if norm(Δρ) < tol
        return [0.5, 0.5]
    end

    eigen_decomp = eigen(Hermitian(Δρ))
    eigenvalues  = eigen_decomp.values
    eigenvectors = eigen_decomp.vectors

    Π1 = zeros(ComplexF64, d, d)
    for i in eachindex(eigenvalues)
        if eigenvalues[i] > tol
            v   = eigenvectors[:, i]
            Π1 += v * v'
        end
    end
    Π2 = I(d) - Π1

    p1 = real(tr(Π1 * ρ_test))
    p2 = real(tr(Π2 * ρ_test))

    # Numerical sanity: p1 + p2 should be 1
    total = p1 + p2
    return [p1/total, p2/total]
end

function update_noise_history!(noise_obj)
  M_petz, M_noise = noise_obj.supermap_petz, noise_obj.supermap_noise
  noise_obj.supermap = M_noise * M_petz * noise_obj.supermap
end

function measure_ancilla(η, rho, noise_options, sigma, rng)
  ancilla_options = []
  for option in noise_options
    _model = CollisionModel(option.supermap, sigma)
    _, _η = apply_collision(_model, rho)
    push!(ancilla_options, _η)
  end

  weights = discrimin(η, ancilla_options...)
  @debug "Discrimination weights: $weights"
  povm = sample(rng, [1, 2], Weights(weights))
  return povm
end

function update_noise_guess(povm, c1, c2, noise_options)
  # update the count of noise choices
  if povm == 1
    c1 += 1
  else
    c2 += 1
  end

  if c1 > c2 || 10*(c1 - c2) == (povm - 2)
    # Assume the first noise model
    noise_guess = noise_options[1]
  elseif c2 > c1 || 10*(c1 - c2) == (povm - 1)
    # Assume the second noise model
    noise_guess = noise_options[2]
  end
  return noise_guess, c1, c2
end

function step_recovery!(step::Int, state::RecoveryState, config::RecoveryConfig, logs::RecoveryLogs)
  # 0. Rename variables for readability
  Ox = config.real_noise.supermap_noise
  O1 = state.noise_options[1].supermap_noise
  O2 = state.noise_options[2].supermap_noise
  Nx = config.real_noise.supermap
  N1 = state.noise_options[1].supermap
  N2 = state.noise_options[2].supermap


  # 1. Apply noise (create intermediate rho_ before recovery)
  ρ_free = unvec((Ox)^step * vec(state.ρ0))
  ρ_rec_ = unvec(Nx * vec(state.ρ0))
  ρ1 = unvec(N1 * vec(state.ρ0))
  ρ2 = unvec(N2 * vec(state.ρ0))
  
  # 2. Recovery
  model = CollisionModel(state.choice.current == 1 ? N1 : N2, config.sigma)
  P = kraus_to_superop(model.kraus_rec);

  # Get the composite state
  ρ_rec = apply_collision(model, ρ_rec_; trace=false)
  model1 = CollisionModel(N1, config.sigma)
  ρ1 = apply_collision(model1, ρ1; trace=false)
  model2 = CollisionModel(N2, config.sigma)
  ρ2 = apply_collision(model2, ρ2; trace=false)

  # 2a. Apply noise again only on the system
  ρ_rec = apply_extended_channel(ρ_rec, Ox, 2^config.n_qubits)
  ρ1 = apply_extended_channel(ρ1, O1, 2^config.n_qubits)
  ρ2 = apply_extended_channel(ρ2, O2, 2^config.n_qubits)

  # 3. Logging
  fid_initial = fidelity(state.ρ0, ρ_rec)
  fid_track = fidelity(state.ρ0, ρ_free)

  @debug "Fidelity wrt initial state: " fid_initial
  @debug "Fidelity of free evolution: " fid_track
  
  push!(logs.ref_fidelities, fid_track)
  push!(logs.fidelities, fid_initial)
  
  # 4. Update noises for the next iteration
  push!(
    logs.maps, 
    (Nx=copy(Nx), N1=copy(N1), N2=copy(N2), P=copy(P)))  # save superoperators at current step
  config.real_noise.supermap = Ox * P * Nx
  state.noise_options[1].supermap = O1 * P * N1
  state.noise_options[2].supermap = O2 * P * N2
  
  # 5. Compare output ancillas
	# First, we extend the ancillas dimensionality
	max_d_ancillas = 4^config.n_qubits
	η, η1, η2 = [embed_state(ancilla, max_d_ancillas) for ancilla in (η, η1, η2)]
	# and normalize them with the probabilities of their respective noise channels
	η1 = state.noise_options[1].probability * η1
	η2 = state.noise_options[2].probability * η2
  w = discrimin(η, η1, η2)
  @debug "Probabilities of the POVM outputs: $w"

  povm = sample(config.rng, [1, 2], Weights(w))
  @debug "Measurement result: $povm"

  # 7. Update guess for the next iteration
  if povm == 1
      append!(state.choice.c1, 1)
      append!(state.choice.c2, 0)
      state.choice.c1_count += 1
  elseif povm == 2
      append!(state.choice.c1, 0)
      append!(state.choice.c2, 1)
      state.choice.c2_count += 1
  end

  inertia = 1
  diff = state.choice.c1_count - state.choice.c2_count

  # The previous-step choice changes only if there is enough inertia in the opposing choice
  # Positive `diff` means that c1 is leading, negative means that c2 is leading
  if state.choice.current == 1
      if diff <= -inertia
          state.choice.current = 2
      end
  else
      if diff >= inertia
          state.choice.current = 1
      end
  end
	append!(state.choice.current_choices, state.choice.current)

  @debug "Updated choice: " new_choice=state.choice.current

  return nothing 
end