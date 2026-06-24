using LinearAlgebra
using StatsBase

function discrimin(ρ_test, ρ1, ρ2, ds, da, q1::Real=0.5, q2::Real=0.5, tol=1e-10)
  size(ρ_test) == (ds * da, ds * da) || error("ρ_test has incompatible dimensions")
  size(ρ1) == (ds * da, ds * da) || error("ρ1 has incompatible dimensions")
  size(ρ2) == (ds * da, ds * da) || error("ρ2 has incompatible dimensions")

  # Trace out the system to get the reduced states of the ancilla
  η_test = ptrace_sys(ρ_test, ds, da)
  η1 = ptrace_sys(ρ1, ds, da)
  η2 = ptrace_sys(ρ2, ds, da)

  Δη = q1 * η1 - q2 * η2

  # Early exit: states are indistinguishable, return uniform
  if norm(Δη) < tol
    return [0.5, 0.5], [0.5*I(da), 0.5*I(da)]
  end

  eigen_decomp = eigen(Hermitian(Δη))
  eigenvalues = eigen_decomp.values
  eigenvectors = eigen_decomp.vectors

  Π1 = zeros(ComplexF64, da, da)
  for i in eachindex(eigenvalues)
    if eigenvalues[i] > tol
      v = eigenvectors[:, i]
      Π1 += v * v'
    end
  end
  Π2 = I(da) - Π1

  p1 = real(tr(Π1 * η_test))
  p2 = real(tr(Π2 * η_test))

  # Numerical sanity: p1 + p2 should be 1
  total = p1 + p2
  return [p1/total, p2/total], [Π1, Π2]
end

function collapse_state(ρ_SA, Π)
  da = size(Π, 1)
  ds = size(ρ_SA, 1) ÷ da
  M = kron(I(ds), Π)
  ρ_post = M * ρ_SA * M'
  Z = max(real(tr(ρ_post)), 1e-12)
  return ρ_post / Z
end

function collapse_map(input_state, output_state; pin=true, tol=1e-12)
  d = size(input_state, 1)
  superop = zeros(ComplexF64, d^2, d^2)

  if pin
    if real(tr(output_state)) <= tol
      return superop
    end
    kraus = get_pin_operators(output_state)
    superop = kraus_to_superop(kraus)
    return superop
  end

  function stochastic_projection(states_in, states_out)
    return states_out * states_in'
  end

  eigen_decomp = eigen(Hermitian(input_state), sortby=x -> -real(x))
  p = clean_eigenvalues(eigen_decomp.values)
  psi_in = eigen_decomp.vectors

  eigen_decomp = eigen(Hermitian(output_state), sortby=x -> -real(x))
  q = clean_eigenvalues(eigen_decomp.values)
  phi_out = eigen_decomp.vectors

  # @debug "Stochastic projection probabilities" p q

  basis = Matrix{ComplexF64}(I, d, d)
  U1 = stochastic_projection(psi_in, basis)
  superop .+= kron(conj(U1), U1)

  T = stochastic_transition(p, q)
  K_transfer = kraus_from_transition(T)
  superop *= kraus_to_superop(K_transfer)

  U2 = stochastic_projection(basis, phi_out)
  superop *= kron(conj(U2), U2)
  return superop
end

function update_noise_guess!(state::RecoveryState, povm::Int)
  if povm == 1
    push!(state.choice.c1, 1)
    push!(state.choice.c2, 0)
    state.choice.c1_count += 1
  elseif povm == 2
    push!(state.choice.c1, 0)
    push!(state.choice.c2, 1)
    state.choice.c2_count += 1
  else
    error("Invalid POVM outcome: $povm")
  end

  if state.choice.c1_count > state.choice.c2_count
    state.choice.current = 1
  elseif state.choice.c2_count > state.choice.c1_count
    state.choice.current = 2
  else
    state.choice.current = povm
  end
  push!(state.choice.history, state.choice.current)
  return nothing

end


function project_to_codespace(rho, n_qubits; real_noise_name="", projection="auto")
  dim = 2^n_qubits

  if projection === "none"
    return rho, 1.0
  end

  P = zeros(ComplexF64, dim, dim)
  projection_name = projection == "auto" ? (
    real_noise_name == "bitflip" || real_noise_name == "phase_damping" ? "xy" : "00_11"
  ) : projection

  if projection_name == "00_11"
    # For amplitude damping, the code space is spanned by |00> and |11>
    P[1, 1] = 1.0   # |0...0><0...0|
    P[dim, dim] = 1.0   # |1...1><1...1|
  elseif projection_name == "xy"
    # For bitflip and phase damping, the code space is spanned by |01> and |10>
    P[2, 2] = 1.0   # |01><01|
    P[3, 3] = 1.0   # |10><10|
  else
    error("Unknown codespace projection: $projection. Use \"auto\", \"00_11\", or \"xy\".")
  end

  rho_proj = P * rho * P
  prob = real(tr(rho_proj))

  prob > 0 || throw(ArgumentError("state has zero overlap with the code space"))

  return rho_proj / prob, prob
end


function iterate_recovery!(
  step::Int,
  state::RecoveryState,
  config::RecoveryConfig,
  logs::RecoveryLogs;
)
  length(state.noise_options) == 2 || error("This workflow currently supports exactly 2 noise options")

  Ox = config.real_noise.supermap_noise
  if step == 1 && !config.start_with_noise
    Nx = I(size(config.real_noise.supermap)[1])
    N1 = I(size(config.real_noise.supermap)[1])
    N2 = I(size(config.real_noise.supermap)[1])
  else
    Nx = config.real_noise.supermap
    N1 = state.noise_options[1].supermap
    N2 = state.noise_options[2].supermap
  end

  ds = 2^config.n_qubits

  # ======================================
  # Step 1: Apply noise only on the reference state to track the fidelity without recovery
  rho_free = unvec((Ox)^step * vec(state.ρ0))
  rho_to_rec = unvec(Nx * vec(state.ρ0))
  rho1 = unvec(N1 * vec(state.ρ0))
  rho2 = unvec(N2 * vec(state.ρ0))

  # ======================================
  # Step 2: Informative collision
  η = config.ancilla_state
  da = size(η, 1)
  model = CollisionModel(config.collision_unitary, config.sigma, ds, da, ancilla_state=η)
  real_kraus = config.real_noise.extended_kraus
  option_kraus = [noise.extended_kraus for noise in state.noise_options]

  # Get the composite state system+ancilla
  rho_to_rec_ = apply_collision(model, rho_to_rec; ancilla_state=η, trace=false)
  rho1_ = apply_collision(model, rho1; ancilla_state=η, trace=false)
  rho2_ = apply_collision(model, rho2; ancilla_state=η, trace=false)

  # Apply noise only to the system
  n_subsystems = config.correlated_noise ? 1 : config.n_qubits
  rho_to_rec_ = apply_extended_channel(rho_to_rec_, real_kraus; dim_to_extend=da)
  rho1_ = apply_extended_channel(rho1_, option_kraus[1]; dim_to_extend=da)
  rho2_ = apply_extended_channel(rho2_, option_kraus[2]; dim_to_extend=da)

  # Create the supermap corresponding to the collision followed by the noise
  Xi = kraus_to_superop(compose_kraus(real_kraus, model.kraus_fwd))
  Xi1 = kraus_to_superop(compose_kraus(option_kraus[1], model.kraus_fwd))
  Xi2 = kraus_to_superop(compose_kraus(option_kraus[2], model.kraus_fwd))

  # ======================================
  # Step 3: Measurement
  q1 = state.noise_options[1].probability
  q2 = state.noise_options[2].probability
  w, Πs = discrimin(rho_to_rec_, rho1_, rho2_, ds, da, q1, q2)
  povm = sample(config.rng, [1, 2], Weights(w))

  # Collapse the system and trace out the ancilla
  if config.nondestructive_measurement
    rho_to_rec = ptrace_ancilla(rho_to_rec_, ds, da)
    rho1 = ptrace_ancilla(rho1_, ds, da)
    rho2 = ptrace_ancilla(rho2_, ds, da)
  else
    rho_to_rec = ptrace_ancilla(collapse_state(rho_to_rec_, Πs[povm]), ds, da)
    rho1 = ptrace_ancilla(collapse_state(rho1_, Πs[povm]), ds, da)
    rho2 = ptrace_ancilla(collapse_state(rho2_, Πs[povm]), ds, da)
    # Get the CPTP map corresponding to the collapse
    Cx = collapse_map(ptrace_ancilla(rho_to_rec_, ds, da), rho_to_rec; pin=config.pin)
    C1 = collapse_map(ptrace_ancilla(rho1_, ds, da), rho1; pin=config.pin)
    C2 = collapse_map(ptrace_ancilla(rho2_, ds, da), rho2; pin=config.pin)
  end

  if config.recover_all
    if config.nondestructive_measurement
      Nx = Xi * Nx
      N1 = Xi1 * N1
      N2 = Xi2 * N2
    else
      Nx = Cx * Xi * Nx
      N1 = C1 * Xi1 * N1
      N2 = C2 * Xi2 * N2
    end
  end

  # ======================================
  # Step 4: Update noise guess and recovery map
  update_noise_guess!(state, povm)
  if config.recover_all
    model = CollisionModel(state.choice.current == 1 ?
                           N1 :
                           N2,
      config.sigma)
  else
    model = CollisionModel(state.choice.current == 1 ?
                           state.noise_options[1].supermap_noise ^ step :
                           state.noise_options[2].supermap_noise ^ step,
      config.sigma)
  end
  P = kraus_to_superop(model.kraus_rec)

  # ======================================
  # Step 5: Recovery
  rho_rec, _ = apply_collision(model, rho_to_rec; trace=true)
  prob_codespace = 1.0
  prob_codespace_free = 1.0
  prob_codespace_initial = 1.0
  rho_initial = if config.codespace_projection === "none"
    state.ρ0
  else
    state.ρ_codespace0
  end
  # if step == 1 || step == config.n_timesteps
  #   @debug "Rho initial before projection:\n$rho_initial" rho_initial
  #   @debug "Rho recovered before projection:\n$rho_rec" rho_rec
  # end
  if config.n_qubits > 1
    rho_rec, prob_codespace = project_to_codespace(
      rho_rec, config.n_qubits;
      real_noise_name=config.real_noise.name,
      projection=config.codespace_projection
    )
    rho_free, prob_codespace_free = project_to_codespace(
      rho_free, config.n_qubits;
      real_noise_name=config.real_noise.name,
      projection=config.codespace_projection
    )
    rho_initial, prob_codespace_initial = project_to_codespace(
      rho_initial, config.n_qubits;
      real_noise_name=config.real_noise.name,
      projection=config.codespace_projection
    )
  end
  # if step == 1 || step == config.n_timesteps
  #   @debug "Rho initial after projection:\n$rho_initial"
  #   @debug "Recovered rho:\n$rho_rec" rho_rec
  #   # @debug "Free rho" rho_free
  # end
  # rho_initial = state.ρ0
  fid_initial = fidelity(rho_initial, rho_rec)
  fid_track = fidelity(rho_initial, rho_free)

  # ======================================
  # Step 6: Updates for the next iteration
  # Save superoperators at current step
  push!(logs.maps, (
    Nx=copy(Nx), N1=copy(N1), N2=copy(N2),
    P=copy(P),
    Cx=copy(Cx),  # Placeholder for collapse map, not used in this version
    Xi=copy(Xi)
  ))
  push!(logs.ref_fidelities, fid_track)
  push!(logs.fidelities, fid_initial)
  push!(logs.choice_history, state.choice.current)
  push!(logs.codespace_overlaps[1], prob_codespace)
  push!(logs.codespace_overlaps[2], prob_codespace_free)
  push!(logs.codespace_overlaps[3], prob_codespace_initial)

  # Update the noise options for the next iteration, 
  #  with the past Petz recovery applied 
  #  and the next noise to be applied
  if config.recover_all
    config.real_noise.supermap = P * Nx
    state.noise_options[1].supermap = P * N1
    state.noise_options[2].supermap = P * N2
  else
    config.real_noise.supermap = Nx
    state.noise_options[1].supermap = N1
    state.noise_options[2].supermap = N2
  end

  return nothing
end

const step_recovery! = iterate_recovery!
