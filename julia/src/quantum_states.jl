"""
    fullrank_state(n_qubits)
Create a full-rank density matrix for n_qubits
"""
function fullrank_state(n_qubits)
    Random.seed!(1234)
    coeffs = normalize(randn(Float64, 2^n_qubits))
    println(coeffs)
    r = zeros(ComplexF64, 2^n_qubits, 2^n_qubits)
    for (i, c) in enumerate(coeffs)
        psi = zeros(ComplexF64, 2^n_qubits)
        psi[i] = c
        r += psi * psi'
    end
    return r
end

"""
    thermal_state(n, beta)
Create the thermal state used to initialize the Petz map
"""
function thermal_state(n, beta)
    σz = [1.0+0.0im 0; 0 -1]
    q = zeros(ComplexF64, 2^n, 2^n)
    for i in 1:n
        for j in 1:i-1
            σz_i = foldl(kron, [k == i ? σz : I2 for k in 1:n])
            σz_j = foldl(kron, [k == j ? σz : I2 for k in 1:n])
            q += σz_i * σz_j
        end
    end
    return exp(beta * q) / tr(exp(beta * q))

end

"""
    input_state(n, a, b)

Create a random input state for n qubits of the form a|00...0> + b|11...1>
"""
function input_state(n, a, b)
    # Ground and excited states of one qubit
    g0 = [0.0 + 0.0im; 1.0]
    e1 = [1.0 + 0.0im; 0.0]
    
    ground = foldl(kron, [g0 for _ in 1:n])
    excited = foldl(kron, [e1 for _ in 1:n])
    state = normalize(a * ground + b * excited)
    return state
end

"""
    codespace_state(n_qubits, a, b)

Create a logic qubit a|00...0> + b|11...1>,
where we adopt the convention that |00...0> is at index 1
of the 2^n_qubits state vector and |11...1> is at index n_qubits
"""
function codespace_state(n_qubits, a, b, c, d)
    psi = zeros(ComplexF64, 2^n_qubits)
    psi[1] = a
    psi[2] = c
    psi[4] = d
    psi[end] = b
    psi = normalize(psi)
    return psi * psi'
end

"""
    random_state(n_qubits)
Sample uniformly in the sphere
"""
function random_state(n_qubits; seed=nothing)
    rng = isnothing(seed) ? Random.default_rng() : Xoshiro(seed)
    ϕ = 2 * rand(rng) * π
    z = 2 * rand(rng) - 1

    θ = acos(z)

    α = cos(θ / 2)
    β = exp(im * ϕ) * sin(θ / 2)

    return input_state(n_qubits, α, β)
end
