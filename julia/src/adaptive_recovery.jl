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
    M1 = (1 / sqrt(2)) * Matrix{ComplexF64}(I, da, da)
    M2 = (1 / sqrt(2)) * Matrix{ComplexF64}(I, da, da)
    return [0.5, 0.5], [M1, M2]
  end

  eigen_decomp = eigen(Hermitian(Δη))
  eigenvalues  = eigen_decomp.values
  eigenvectors = eigen_decomp.vectors

  Π1 = zeros(ComplexF64, da, da)
  for i in eachindex(eigenvalues)
      if eigenvalues[i] > tol
          v   = eigenvectors[:, i]
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

function collapse_map(input_state, output_state)
  d = size(input_state, 1)
  superop = zeros(ComplexF64, d^2, d^2)

  function stochastic_projection(states_in, states_out)
    return states_out * states_in'
  end

  eigen_decomp = eigen(Hermitian(input_state), sortby = x -> -real(x))
  p            = clean_eigenvalues(eigen_decomp.values)
  psi_in       = eigen_decomp.vectors

  eigen_decomp = eigen(Hermitian(output_state), sortby = x -> -real(x))
  q            = clean_eigenvalues(eigen_decomp.values)
  phi_out      = eigen_decomp.vectors

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
    state.choice.current = state.choice.history[end]
  end
  push!(state.choice.history, state.choice.current)
  return nothing

end

function iterate_recovery!(step::Int, state::RecoveryState, config::RecoveryConfig, logs::RecoveryLogs)
  length(state.noise_options) == 2 || error("This workflow currently supports exactly 2 noise options")

  Ox = config.real_noise.supermap_noise
  O1 = state.noise_options[1].supermap_noise
  O2 = state.noise_options[2].supermap_noise
  Nx = config.real_noise.supermap
  N1 = state.noise_options[1].supermap
  N2 = state.noise_options[2].supermap

  maps_to_log = (Nx=copy(Nx), N1=copy(N1), N2=copy(N2))

  ds = 2^config.n_qubits

  rho_free = unvec((Ox)^step * vec(state.ρ0))
  rho_to_rec = unvec(Nx * vec(state.ρ0))
  rho1 = unvec(N1 * vec(state.ρ0))
  rho2 = unvec(N2 * vec(state.ρ0))

  η = config.ancilla_state
  da = size(η, 1)
  model = CollisionModel(config.collision_unitary, config.sigma, ds, da, ancilla_state=η)
  real_kraus = config.real_noise.extended_kraus
  option_kraus = [noise.extended_kraus for noise in state.noise_options]

  rho_to_rec_ = apply_collision(model, rho_to_rec; ancilla_state=η, trace=false)
  rho1_ = apply_collision(model, rho1; ancilla_state=η, trace=false)
  rho2_ = apply_collision(model, rho2; ancilla_state=η, trace=false)

  n_subsystems = config.correlated_noise ? 1 : config.n_qubits
  rho_to_rec_ = apply_channel(real_kraus, rho_to_rec_, n_subsystems; extra_dims=da)
  rho1_ = apply_channel(option_kraus[1], rho1_, n_subsystems; extra_dims=da)
  rho2_ = apply_channel(option_kraus[2], rho2_, n_subsystems; extra_dims=da)

  Xi = kraus_to_superop(compose_kraus(real_kraus, model.kraus_fwd))
  Xi1 = kraus_to_superop(compose_kraus(option_kraus[1], model.kraus_fwd))
  Xi2 = kraus_to_superop(compose_kraus(option_kraus[2], model.kraus_fwd))

  q1 = state.noise_options[1].probability
  q2 = state.noise_options[2].probability
  w, Πs = discrimin(rho_to_rec_, rho1_, rho2_, ds, da, q1, q2)
  @debug "Discrimination probabilities" w
  povm = sample(config.rng, [1, 2], Weights(w))

  rho_to_rec = ptrace_ancilla(collapse_state(rho_to_rec_, Πs[povm]), ds, da)
  rho1 = ptrace_ancilla(collapse_state(rho1_, Πs[povm]), ds, da)
  rho2 = ptrace_ancilla(collapse_state(rho2_, Πs[povm]), ds, da)

  Cx = collapse_map(ptrace_ancilla(rho_to_rec_, ds, da), rho_to_rec)
  C1 = collapse_map(ptrace_ancilla(rho1_, ds, da), rho1)
  C2 = collapse_map(ptrace_ancilla(rho2_, ds, da), rho2)

  Nx = Cx * Xi * Nx
  N1 = C1 * Xi1 * N1
  N2 = C2 * Xi2 * N2

  update_noise_guess!(state, povm)
  model = CollisionModel(state.choice.current == 1 ? N1 : N2, config.sigma)
  P = kraus_to_superop(model.kraus_rec)

  rho_rec, _ = apply_collision(model, rho_to_rec; trace=true)
  rho_rec_1, _ = apply_collision(model1, rho_to_rec; trace=true)
  rho_rec_2, _ = apply_collision(model2, rho_to_rec; trace=true)

  @debug "F if recover 1 = " fidelity(state.ρ0, rho_rec_1)
  @debug "F if recover 2 = " fidelity(state.ρ0, rho_rec_2)

  fid_initial = fidelity(state.ρ0, rho_rec)
  fid_track = fidelity(state.ρ0, rho_free)

  maps_to_log = (; 
    maps_to_log..., 
    Xi=copy(Xi), 
    Xi1=copy(Xi1), 
    Xi2=copy(Xi2), 
    Cx=copy(Cx), 
    C1=copy(C1), 
    C2=copy(C2), 
    P=copy(P)
  )
  push!(logs.maps, maps_to_log)
  push!(logs.ref_fidelities, fid_track)
  push!(logs.fidelities, fid_initial)
  push!(logs.choice_history, state.choice.current)

  # Update the noise options for the next iteration, 
  #  with the past Petz recovery applied 
  #  and the next noise to be applied
  config.real_noise.supermap = Ox * P * Nx
  state.noise_options[1].supermap = O1 * P * N1
  state.noise_options[2].supermap = O2 * P * N2

  @debug "Current choice" state.choice.current

  return nothing
end

const step_recovery! = iterate_recovery!
