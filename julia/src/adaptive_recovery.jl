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
function discrimin(ρ_test, ρ1, ρ2)
    # Compute the difference of density matrices
    Δρ = ρ1 - ρ2
    
    # Diagonalize to find eigenvalues and eigenvectors
    eigen_decomp = eigen(Hermitian(Δρ))
    eigenvalues = eigen_decomp.values
    eigenvectors = eigen_decomp.vectors
    
    # Construct Π₁: projector onto positive eigenspace
    Π1 = zeros(ComplexF64, size(ρ1))
    for i in eachindex(eigenvalues)
        if eigenvalues[i] > 1e-10  # positive eigenvalue (with numerical tolerance)
            v = eigenvectors[:, i]
            Π1 += v * v'  # Add rank-1 projector |v⟩⟨v|
        end
    end
    
    # Construct Π₂: complement projector
    Π2 = I - Π1
    
    # Compute probabilities
    p1 = real(tr(Π1 * ρ_test))
    p2 = real(tr(Π2 * ρ_test))
    
    return [p1, p2]
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

    # 0. Setup models
    M_petz = state.noise_guess.supermap_petz
    M_noise = state.noise_guess.supermap_noise
    # Update M_total
    if step > 1
        state.M_total = (M_noise * M_petz) * state.M_total
    end
    model = CollisionModel(state.M_total, config.sigma)
    
    # 1. Apply noise (update rho1 and create intermediate rho2_)
    state.ρ_free = apply_channel(config.real_noise.kraus, state.ρ_free, config.n_qubits)
    ρ_rec_ = apply_channel(config.real_noise.kraus, state.ρ_rec, config.n_qubits)
    
    # 2. Recovery
    # Update rho2 in the state struct
    state.ρ_rec, η = apply_collision(model, ρ_rec_)
    
    # 3. Logging
    fid_initial = fidelity(state.ρ0, state.ρ_rec)
    fid_ref = fidelity(config.sigma, state.ρ_rec)
    fid_track = fidelity(state.ρ0, state.ρ_free)

    @debug "Fidelity wrt initial state: $fid_initial"
    @debug "Fidelity wrt reference state: $fid_ref"
    @debug "Fidelity of free evolution: $fid_track"
    
    push!(logs.ref_fidelities, fid_track)
    push!(logs.fidelities, fid_initial)
    
    # 4. Measure Ancilla and Update Guess
    if step > 1
        for option in state.noise_options
            update_noise_history!(option)
        end
    end
    povm = measure_ancilla(η, ρ_rec_, state.noise_options, config.sigma, config.rng)
    @debug "Measure output: $povm"
    
    # Update the remaining state variables
    state.noise_guess, state.c1, state.c2 = update_noise_guess(
        povm, state.c1, state.c2, state.noise_options)
    @debug "Updated noise guess:\t$(state.noise_guess.name)"

    return nothing 
end