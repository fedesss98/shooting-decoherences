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

using LinearAlgebra

function _swap_unitary(ds::Int, da::Int)
    U = zeros(ComplexF64, ds * da, ds * da)
    for s in 1:ds
        for a in 1:da
            row = (a - 1) * ds + s
            col = (s - 1) * da + a
            U[row, col] = 1.0 + 0.0im
        end
    end
    return U
end

function _n_qubit_exchange_unitary(n_qubits::Int, g::Float64=0.1, t::Float64=1.0)
    # Qubit raising and lowering operators
    sp = [0.0 1.0; 0.0 0.0]
    sm = [0.0 0.0; 1.0 0.0]

    # Total dimension, considering 1 qubit ancilla
    n_total = n_qubits + 1
    d = 2^(n_total)
    H_int = zeros(ComplexF64, d, d)

    # The ancilla is the 'last system' in the kronecker product
    ancilla_idx = n_total
    sp_anc = embed_operator(sp, ancilla_idx, n_total)
    sm_anc = embed_operator(sm, ancilla_idx, n_total)

    # Sum over all k system qubits
    for k in 1:n_qubits
        sp_k = embed_operator(sp, k, n_total)
        sm_k = embed_operator(sm, k, n_total)

        exchange_term = (sp_anc * sm_k) + (sm_anc * sp_k)

        H_int += (g / n_qubits) * exchange_term
    end

    # Return the time evolution unitary U(t)
    U = exp(-1im * H_int * t)
    return U
end

function kraus_from_unitary(
    U::AbstractMatrix{T},
    d_s::Int,
    d_a::Int;
    ancilla_state::AbstractMatrix,
    atol::Real=1e-12,
) where T
    size(U) == (d_s * d_a, d_s * d_a) || throw(ArgumentError("wrong U size"))
    size(ancilla_state) == (d_a, d_a) || throw(ArgumentError("wrong ancilla size"))

    eig = eigen(Hermitian(Matrix{T}(ancilla_state)))
    vals = eig.values
    vecs = eig.vectors

    kraus = Matrix{T}[]

    # (s,a) -> (s-1)*d_a + a
    for ν in eachindex(vals)
        pν = real(vals[ν])
        pν <= atol && continue
        ψν = vecs[:, ν]

        for i in 1:d_a
            K = zeros(T, d_s, d_s)

            for sout in 1:d_s
                row_base = (sout - 1) * d_a
                for sin in 1:d_s
                    col_base = (sin - 1) * d_a
                    amp = zero(T)
                    for ain in 1:d_a
                        row = row_base + i
                        col = col_base + ain
                        amp += U[row, col] * ψν[ain]
                    end
                    K[sout, sin] = sqrt(T(pν)) * amp
                end
            end

            push!(kraus, K)
        end
    end

    return kraus
end


function compose_kraus(
    kraus2::Vector{<:AbstractMatrix},
    kraus1::Vector{<:AbstractMatrix},
)
    # channel 1 first, then channel 2
    out = Matrix{eltype(kraus1[1])}[]
    for K2 in kraus2
        for K1 in kraus1
            push!(out, K2 * K1)
        end
    end
    return out
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


"""
  get_kraus_operators(noise, gamma, t)
Route to the correct Kraus operators given the name of the noise.
The output is a List of Kraus operators.
"""
function get_kraus_operators(noise, gamma, t; rng=nothing, n_qubits=1)
    if noise == "amplitude_damping"
        return get_amplitudedamping_operators(gamma, t; n_qubits=n_qubits)
    elseif noise == "phase_damping"
        return get_phasedamping_operators(gamma, t; n_qubits=n_qubits)
    elseif noise == "bitflip"
        return get_bitflip_operators(gamma, t; n_qubits=n_qubits)
    elseif noise == "depolarizing"
        return get_depolarizing_operators(gamma, t; n_qubits=n_qubits)
    elseif noise == "random" || noise == "general"
        return get_random_operators(rng; n_qubits=n_qubits)
    else
        error("Unknown noise model: $noise")
    end

end

function unvec(state)
    dims = Int(sqrt(size(state, 1)))

    return reshape(state, dims, dims)
end

"""
  embed_state(ρ, d_target)
Embed a density matrix ρ into a larger Hilbert space of dimension d_target by padding with zeros.
This is used to match the dimensions of the ancilla when performing discrimination or recovery.
"""
function embed_state(ρ, d_target)
    d = size(ρ, 1)
    d == d_target && return ρ
    ρ_out = zeros(ComplexF64, d_target, d_target)
    ρ_out[1:d, 1:d] = ρ
    return ρ_out
end

function embed_operator(op::Matrix, target_index::Int, n::Int)
    I2 = [1.0 0.0; 0.0 1.0] # 2x2 Identity
    # Start the Kronecker product chain
    result = (target_index == 1) ? op : I2
    for i in 2:n
        next_op = (i == target_index) ? op : I2
        result = kron(result, next_op)
    end
    return result
end


function clean_probability_vector(x; tol=1e-12)
    p = Float64.(real.(x))

    p[abs.(p) .< tol] .= 0.0

    if any(p .< -tol)
        error("Invalid probability vector: negative entry $(minimum(p))")
    end

    p .= max.(p, 0.0)

    s = sum(p)
    s > tol || error("Invalid probability vector: total weight is zero")

    return p ./ s
end


function project_zero_marginals!(A)
    m, n = size(A)

    rowA = sum(A, dims=2)
    colA = sum(A, dims=1)
    totalA = sum(A)

    A .= A .- rowA ./ n .- colA ./ m .+ totalA / (m * n)

    return A
end

"""
  stochastic_transition(p, q; A=nothing, tol=1e-12, checktol=1e-8, safety=1e-10)
Given two probability distributions p and q, construct a stochastic transition matrix T such that T*p = q.
This is done by first constructing a transport plan R = q*p' + A, 
where A is an optional adjustment matrix to ensure positivity and stability.
Then, T is obtained by normalizing the columns of R by p. 
If p[i] is zero, the corresponding column of T is set to q (arbitrary stochastic column, 
since it does not affect the result).
The function also includes checks to ensure that R is a valid transport plan
(non-negative and with correct marginals), and that T is a valid stochastic matrix.
"""
function stochastic_transition(p, q; A=nothing, tol=1e-12, checktol=1e-8, safety=1e-10)
    p = clean_probability_vector(p; tol=tol)
    q = clean_probability_vector(q; tol=tol)

    n = length(p)
    m = length(q)

    R0 = q * p'   # always valid

    if A === nothing
        R = copy(R0)
    else
        A = Matrix{Float64}(A)

        size(A) == (m, n) || error("A has size $(size(A)), expected $((m, n))")

        project_zero_marginals!(A)

        # Choose largest α ∈ [0, 1] such that R0 + αA ≥ 0
        negative_A = A .< 0

        if any(negative_A)
            αmax = minimum(R0[negative_A] ./ (-A[negative_A]))
            α = min(1.0, max(0.0, (1.0 - safety) * αmax))
        else
            α = 1.0
        end

        R = R0 + α * A

        # Clean only tiny numerical negatives
        tiny_negative = (R .< 0.0) .& (R .> -tol)
        R[tiny_negative] .= 0.0

        # Absolute fallback: if something still went wrong, use the guaranteed plan
        if any(R .< -tol)
            @warn "A could not be made feasible; falling back to independent transport plan q*p'."
            R = copy(R0)
        end
    end

    colerr = maximum(abs.(vec(sum(R, dims=1)) .- p))
    rowerr = maximum(abs.(vec(sum(R, dims=2)) .- q))

    if colerr > checktol || rowerr > checktol
        @warn "Transport plan lost marginals; falling back to independent transport plan q*p'." colerr rowerr
        R = copy(R0)
    end

    T = zeros(Float64, m, n)

    for i in 1:n
        if p[i] > tol
            T[:, i] = R[:, i] ./ p[i]
        else
            # This column is irrelevant because p[i] = 0
            T[:, i] = q
        end
    end

    # Clean tiny numerical noise
    T[abs.(T) .< tol] .= 0.0
    T .= max.(T, 0.0)

    # Ensure every column is stochastic
    for i in 1:n
        s = sum(T[:, i])
        if s > tol
            T[:, i] ./= s
        else
            T[:, i] .= q
        end
    end

    # Final sanity check
    if maximum(abs.(sum(T, dims=1)[:] .- 1.0)) > checktol
        error("Invalid transition matrix: columns are not normalized.")
    end

    if maximum(abs.(T * p .- q)) > checktol
        error("Invalid transition matrix: T*p is not q.")
    end

    return T
end


"""
  kraus_from_transition(T)
Given a stochastic transition matrix T, construct a set of Kraus operators that implement the corresponding quantum channel. 
Each non-zero entry T[j, i] corresponds to a Kraus operator K_ji = sqrt(T[j, i]) |j⟩⟨i|, 
where |i⟩ and |j⟩ are the standard basis states of the input and output Hilbert spaces, respectively.
"""
function kraus_from_transition(T)
    m, n = size(T)
    Ks = Matrix{Float64}[]

    for j in 1:m
        for i in 1:n
            if T[j, i] > 0
                K = zeros(m, n)
                K[j, i] = sqrt(T[j, i])
                push!(Ks, K)
            end
        end
    end

    return Ks
end

function clean_eigenvalues(eigvals, tol=1e-10)
    cleaned = similar(eigvals)
    for i in eachindex(eigvals)
        val = real(eigvals[i])
        if abs(val) < tol
            cleaned[i] = 0.0
        elseif val < 0
            cleaned[i] = 0.0
        else
            cleaned[i] = val
        end
    end
    return cleaned
end
