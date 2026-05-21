
"""
    CollisionModel

A struct to organize the simulation of a recovery map via a collision model.
It stores the system/ancilla dimensions, the reference state, the forward/recovery Kraus ops,
and the resulting Stinespring dilation Unitary.
"""
struct CollisionModel{T<:Number}
    dim_sys::Int
    dim_anc::Int
    sigma::Matrix{T}              # Reference state
    kraus_fwd::Vector{Matrix{T}}  # Forward noise {K_i}
    kraus_rec::Vector{Matrix{T}}  # Petz recovery {R_i}
    U::Matrix{T}                  # The joint system-ancilla unitary
end

"""
    CollisionModel(kraus_fwd, sigma; n=1)

General constructor. Given arbitrary Kraus operators and a reference state:
1. Computes the Petz Recovery Kraus operators.
2. Constructs the Stinespring Collision Unitary U.
"""
function CollisionModel(kraus_fwd::Vector{Matrix{T1}}, sigma::Matrix{T2}; n::Int=1) where {T1<:Number, T2<:Number}
    # Determine a common type
    T = promote_type(T1, T2)
    # and convert inputs to this common type
    sigma = Matrix{T}(sigma)
    kraus_fwd = Matrix{T}[Matrix{T}(K) for K in kraus_fwd]
    # Optionally expand single-qubit Kraus to n-qubits if needed
    if n > 1
        expand_kraus_operators!(kraus_fwd, n)
    end
    d_sys = size(sigma, 1)
    d_anc = length(kraus_fwd) # Ancilla dimension = number of Kraus ops
    
    # 1. Compute Forward Action on Sigma: N(σ) = ∑ K σ K'
    N_sigma = sum(K * sigma * K' for K in kraus_fwd)

    # 2. Compute Petz Recovery Kraus Ops: R_i = σ^(1/2) K_i' N(σ)^(-1/2)
    # Note: Using pinv for stability, assuming Hermitian matrices
    σ_sqrt = sqrt(Hermitian(sigma))
    σ_out_inv_sqrt = inv(sqrt(Hermitian(N_sigma + 1e-10*I)))
    kraus_rec = [σ_sqrt * K' * σ_out_inv_sqrt for K in kraus_fwd]

    # 3. Construct Stinespring Unitary U
    U = unitary_dilation(kraus_rec, d_sys, d_anc)

    return CollisionModel(d_sys, d_anc, sigma, kraus_fwd, kraus_rec, U)
end


"""
    CollisionModel(noise_channel::Function, gamma, sigma; nsteps=100)

Constructor for when Kraus operators are unknown. The noise channel is given as
a function `noise_channel(rho, gamma)` that returns the output state.

The Petz recovery is constructed by numerical Choi-Jamiolkowski:
- Discretize gamma ∈ [0, gamma] into `nsteps`
- Compute dΦ/dγ ≈ (Φ(γ+dγ) - Φ(γ))/dγ
- Extract Kraus operators via eigendecomposition of Choi matrix
"""
function CollisionModel(noise_channel::Function, gamma::Real, sigma::Matrix{T}; 
                            nsteps::Int=100) where T<:Number
    d = size(sigma, 1)
    
    # Maximize basis for Choi matrix
    max_ent = sum(kron(Matrix{T}(I, d, d)[:, i:i], Matrix{T}(I, d, d)[:, i:i]) 
                  for i in 1:d) / sqrt(d)
    rho_max = max_ent * max_ent'
    
    # Apply channel to maximally entangled state
    # Φ ⊗ I acting on |Ψ⟩⟨Ψ| gives Choi matrix
    function choi_matrix(γ)
        choi = zeros(T, d*d, d*d)
        for i in 1:d, j in 1:d
            input = zeros(T, d, d)
            input[i, j] = one(T)
            output = noise_channel(input, γ)
            for k in 1:d, l in 1:d
                choi[(i-1)*d+k, (j-1)*d+l] = output[k, l]
            end
        end
        return choi
    end
    
    # Derivative at gamma
    dγ = gamma / nsteps
    J = (choi_matrix(gamma + dγ) - choi_matrix(gamma)) / dγ
    J = (J + J') / 2  # Symmetrize
    
    # Extract Kraus from Choi eigendecomposition
    vals, vecs = eigen(Hermitian(J))
    kraus_fwd = Matrix{T}[]
    for i in 1:d*d
        if real(vals[i]) > 1e-12
            K = sqrt(abs(vals[i])) * reshape(vecs[:, i], d, d)
            push!(kraus_fwd, K)
        end
    end
    
    isempty(kraus_fwd) && push!(kraus_fwd, zeros(T, d, d))
    
    CollisionModel(kraus_fwd, sigma)
end

"""
    CollisionModel(M::Matrix, sigma::Matrix)

Constructor from superoperator M (d²×d² matrix in column-stacking convention).
Extracts Kraus operators via Choi matrix eigendecomposition.
"""
function CollisionModel(M::Matrix{T}, sigma::Matrix{T}) where T<:Number
    d = size(sigma, 1)
    
    # Choi matrix from superoperator: reshape and transpose
    # J = sum_ij |i⟩⟨j| ⊗ M(|i⟩⟨j|)
    choi = reshape(M, d, d, d, d)
    choi = permutedims(choi, [1, 3, 2, 4])
    choi = reshape(choi, d*d, d*d)
    
    # Symmetrize
    choi = (choi + choi') / 2
    
    # Extract Kraus operators
    vals, vecs = eigen(Hermitian(choi))
    kraus_fwd = Matrix{T}[]
    for i in 1:d*d
        if real(vals[i]) > 1e-12
            K = sqrt(abs(vals[i])) * reshape(vecs[:, i], d, d)
            push!(kraus_fwd, K)
        end
    end
    
    isempty(kraus_fwd) && push!(kraus_fwd, zeros(T, d, d))
    
    if length(kraus_fwd) > 64
        error("Too many Kraus operators extracted: $(length(kraus_fwd)). Consider increasing the threshold or checking the input superoperator.")
    end

    CollisionModel(kraus_fwd, sigma)
end

"""
    kraus_from_unitary(U, d_s, d_a; ancilla_state=nothing)

Extract Kraus operators for the induced system channel from a joint unitary U
under system ⊗ ancilla ordering.

If `ancilla_state === nothing`, the ancilla input is assumed to be |0><0|.

If `ancilla_state = sum_j p_j |ψ_j><ψ_j|`, then the Kraus operators are
    K_{i,j} = sqrt(p_j) (I ⊗ <i|) U (I ⊗ |ψ_j>)
for ancilla basis states |i>.
"""
function kraus_from_unitary(
    U::AbstractMatrix{T},
    d_s::Int,
    d_a::Int;
    ancilla_state::Union{Nothing,AbstractMatrix}=nothing,
    atol::Real=1e-12,
) where T
    size(U, 1) == d_s * d_a || throw(ArgumentError("U has wrong size"))
    size(U, 2) == d_s * d_a || throw(ArgumentError("U has wrong size"))

    # Default ancilla input: |0><0|
    ρa = ancilla_state === nothing ? ancilla_ground_state(T, d_a) : Matrix{T}(ancilla_state)
    size(ρa) == (d_a, d_a) || throw(ArgumentError("ancilla_state has wrong size"))

    # Spectral decomposition of ancilla state
    eig = eigen(Hermitian(ρa))
    vals = eig.values
    vecs = eig.vectors

    K = Matrix{T}[]

    # System ⊗ ancilla index mapping: (s,a) -> (s-1)*d_a + a
    for j in eachindex(vals)
        pj = real(vals[j])
        pj < -atol && throw(ArgumentError("ancilla_state is not positive semidefinite"))
        pj <= atol && continue

        ψj = vecs[:, j]

        for aout in 1:d_a
            Ki = zeros(T, d_s, d_s)

            for sout in 1:d_s
                row_base = (sout - 1) * d_a
                for sin in 1:d_s
                    col_base = (sin - 1) * d_a

                    amp = zero(T)
                    for ain in 1:d_a
                        row = row_base + aout
                        col = col_base + ain
                        amp += U[row, col] * ψj[ain]
                    end
                    Ki[sout, sin] = sqrt(T(pj)) * amp
                end
            end

            push!(K, Ki)
        end
    end

    return K
end

"""
    CollisionModel(U, sigma, d_sys, d_anc; ancilla_state=nothing, compute_recovery=true)

Construct a collision model from a known joint unitary U.

- `sigma` is the system reference state used for Petz recovery.
- `ancilla_state` is the ancilla input state used to derive the forward Kraus operators.
  If omitted, |0><0| is assumed.
- If `compute_recovery=false`, the recovery Kraus list is left empty.
"""
function CollisionModel(
    U::Matrix{T1},
    sigma::Matrix{T2},
    d_sys::Int,
    d_anc::Int;
    ancilla_state::Union{Nothing,AbstractMatrix}=nothing,
    compute_recovery::Bool=true,
) where {T1<:Number, T2<:Number}

    T = promote_type(T1, T2)

    UT = Matrix{T}(U)
    sigmaT = Matrix{T}(sigma)

    size(UT) == (d_sys * d_anc, d_sys * d_anc) ||
        throw(ArgumentError("U must be of size $(d_sys*d_anc) × $(d_sys*d_anc)"))
    size(sigmaT) == (d_sys, d_sys) ||
        throw(ArgumentError("sigma must be of size $d_sys × $d_sys"))

    kraus_fwd = kraus_from_unitary(UT, d_sys, d_anc; ancilla_state=ancilla_state)

    kraus_rec = Matrix{T}[]
    if compute_recovery
        N_sigma = sum(K * sigmaT * K' for K in kraus_fwd)
        σ_sqrt = sqrt(Hermitian(sigmaT))
        σ_out_inv_sqrt = inv(sqrt(Hermitian(N_sigma + T(1e-10) * I)))
        kraus_rec = [σ_sqrt * K' * σ_out_inv_sqrt for K in kraus_fwd]
    end

    return CollisionModel{T}(d_sys, d_anc, sigmaT, kraus_fwd, kraus_rec, UT)
end


"""
    expand_kraus_operators!(kraus_single::Vector{Matrix{T}}, N::Int) where T

Expand single-qubit Kraus operators to N-qubit system via tensor products.
Modifies the input vector in place, replacing it with all m^N tensor product combinations.
"""
function expand_kraus_operators!(kraus_single::Vector{Matrix{T}}, N::Int) where T    
    d_single = size(kraus_single[1], 1)
    @assert d_single == 2 "Single-qubit Kraus operators must be 2×2"
    
    m = length(kraus_single)
    kraus_expanded = Matrix{T}[]
    
    # Generate all tensor product combinations
    for idx in Iterators.product([1:m for _ in 1:N]...)
        K = kraus_single[idx[1]]
        for i in 2:N
            K = kron(K, kraus_single[idx[i]])
        end
        push!(kraus_expanded, K)
    end
    
    # Modify in place
    empty!(kraus_single)
    append!(kraus_single, kraus_expanded)
    
    return kraus_single
end