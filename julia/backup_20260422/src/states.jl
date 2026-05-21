const I2 = [1.0+0.0im 0.0; 0.0 1.0]
# Ground and excited states of one qubit
const g0 = [0.0 + 0.0im; 1.0]
const e1 = [1.0 + 0.0im; 0.0]


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


function thermal_chain_state(n, beta)
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


function codespace_state(n, a, b)
    ground = foldl(kron, [g0 for _ in 1:n])
    excited = foldl(kron, [e1 for _ in 1:n])
    state = normalize(a * ground + b * excited)
    return state
end


function random_state(n_qubits)
    # Sample uniformly in the sphere
    ϕ = 2 * rand() * π
    z = 2 * rand() - 1

    θ = acos(z)

    α = cos(θ / 2)
    β = exp(im * ϕ) * sin(θ / 2)

    return codespace_state(n_qubits, α, β)
end