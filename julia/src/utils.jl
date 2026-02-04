using TOML
using LinearAlgebra


"""
    verify_unitary(U::Matrix; atol=1e-10)

Verify that a matrix is unitary (U† U = I).
"""
function verify_unitary(U::Matrix{ComplexF64}; atol=1e-10)
  n = size(U, 1)
  return isapprox(U' * U, I(n), atol=atol)
end

"""
    chop_parts(z::ComplexF64)

Set small values of a complex number to zero.
"""
function chop_parts(z::ComplexF64, tol::Float64)
  # Check real part
  r = real(z)
  new_r = abs(r) > tol ? r : 0.0

  # Check imaginary part
  i = imag(z)
  new_i = abs(i) > tol ? i : 0.0

  return complex(new_r, new_i)
end

"""
    chop!(input)

Set small elements to zero.
"""
function chop!(input, tol::Float64=1e-9)
  map!(z -> chop_parts(z, tol), input, input)
end


"""
  partial_trace(ρ, dims, system_to_trace)

Calculates the partial trace of a multipartite density matrix ρ.
"""
function partial_trace(ρ, dims, system_to_trace)
  # Validate dimensions
  if prod(dims) != size(ρ, 1) && prob(dims) != size(ρ, 2)
    throw(DimensionMismatch("Dims do not match matrix size"))
  end

  # Reshape into a tensor
  tensor = reshape(ρ, (dims..., dims...))

  # Identify indices to keep
  N = length(dims)
  trace_idx_row = system_to_trace
  trace_idx_col = system_to_trace + N
  keep_indices = setdiff(1:2N, [trace_idx_row, trace_idx_col])

  # Trace
  perm = [keep_indices; trace_idx_row; trace_idx_col]
  tensor_perm = permutedims(tensor, perm)
  # Reshape to separate the part we keep and the part we trace
  # (Dim_Keep, Dim_Keep, Dim_Trace, Dim_Trace)
  dim_trace = dims[system_to_trace]
  dim_keep_total = div(length(ρ), dim_trace^2)

  matrix_stage = reshape(tensor_perm, (dim_keep_total, dim_trace, dim_trace))

  # Now trace: sum over the diagonal of the last two dimensions
  # effectively: result[i] = sum(matrix_stage[i, k, k] for k)
  rho_reduced = map(i -> tr(matrix_stage[i, :, :]), 1:dim_keep_total)

  # Reshape back to square matrix
  final_dim = Int(sqrt(dim_keep_total))
  return reshape(rho_reduced, (final_dim, final_dim))
end


"""
    partial_trace_ancilla(rho::Matrix, dim_system::Int, dim_ancilla::Int)

Compute the partial trace over the ancilla (second subsystem).

# Arguments
- `rho`: Density matrix of the composite system (dim_system × dim_ancilla square)
- `dim_system`: Dimension of the system to keep (qubit = 2)
- `dim_ancilla`: Dimension of the ancilla to trace out (4-level = 4)

# Returns
- Reduced density matrix of the system (dim_system × dim_system)
"""
function partial_trace_ancilla(rho::Matrix{ComplexF64}, dim_system::Int, dim_ancilla::Int)
  total_dim = dim_system * dim_ancilla

  if size(rho) != (total_dim, total_dim)
    error("Density matrix size doesn't match system dimensions!")
  end

  rho_reduced = zeros(ComplexF64, dim_system, dim_system)

  # Sum over ancilla basis states
  for k in 0:(dim_ancilla-1)
    for i in 0:(dim_system-1)
      for j in 0:(dim_system-1)
        # Map indices: |i⟩_S ⊗ |k⟩_A has index i*dim_ancilla + k + 1 (Julia 1-indexed)
        row_idx = i * dim_ancilla + k + 1
        col_idx = j * dim_ancilla + k + 1

        rho_reduced[i+1, j+1] += rho[row_idx, col_idx]
      end
    end
  end

  return rho_reduced
end

"""
    matrix_power_pseudo(ρ, p; tol)
Compute ρ^p using pseudoinverse for singular matrices.
Zero eigenvalues (< tol) are kept as zero instead of inverted.
"""
function matrix_power_pseudo(ρ, p; tol=1e-10)
  eigen_decomp = eigen(Hermitian(ρ))
  λ = eigen_decomp.values
  V = eigen_decomp.vectors

  # Apply power only to non-zero eigenvalues
  λ_powered = similar(λ, ComplexF64)
  for i in eachindex(λ)
    if abs(λ[i]) > tol
      λ_powered[i] = λ[i]^p
    else
      λ_powered[i] = 0.0
    end
  end

  return V * Diagonal(λ_powered) * V'
end

function read_config(config_path)
  if !isfile(config_path)
    error("Config file not found at: $config_path")
  end

  config = TOML.parsefile(config_path)
  println("""Loaded configuration for experiment $(config["experiment"])""")

  return config
end



"""
    kraus_operators_recovery1(pa::Float, pb::Float, a1, a2, b1, b2)

Construct the Kraus Operator for the recovery map
Λ[ρ] = Tr(ρ) σ,
were the recovered state σ is defined by
pa [a1 a2] + pb[b1 b2] * [b1 b2]
"""
function kraus_operators_recovery1(pa::Float64, pb::Float64, a1::Float64, a2::Float64, b1::Float64, b2::Float64)
    K1 = sqrt(pa) * [a1 0; a2 0]
    K2 = sqrt(pb) * [b1 0; b2 0]
    K3 = sqrt(pa) * [0 a1; 0 a2]
    K4 = sqrt(pb) * [0 b1; 0 b2]

    return [K1, K2, K3, K4]
end

"""
    isometry(system_state::Vector)

Implement the action of the isomery V, acting on a state of the system
and extending the action of the map in the system+ancilla Hilbert Space.
"""
function isometry(system_state::Vector, ancilla_basis, kraus_operators)
    expanded_dims = size(system_state)[1] * size(ancilla_basis[1])[1]
    result = Vector{ComplexF64}(undef, expanded_dims)
    for (k, a) in zip(kraus_operators, ancilla_basis)
        result += kron(k * system_state, a)
    end
    return result
end


"""
Enforces physical validity: Hermiticity and Normalization.
Removes imaginary noise from diagonal and resets Trace to 1.
"""
function enforce_physical!(rho::Matrix{ComplexF64})
    # 1. Symmetrize to remove imaginary drift (force Hermiticity)
    rho .= (rho .+ rho') ./ 2
    
    # 2. Normalize Trace (fix Petz contraction)
    tr_val = real(tr(rho))
    if tr_val > 1e-12
        rho ./= tr_val
    end
    return rho
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
  build_superoperators(model)
Builds the superoperator matrix which implements the n+1 evolution step:
collision + noise
"""
function build_superoperators(model)
  M_petz = kraus_to_superop(model.kraus_rec)
  M_noise = kraus_to_superop(model.kraus_fwd)

  return M_petz, M_noise
end