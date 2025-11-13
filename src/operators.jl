"""
    density_matrix(a)

a ∈ [-1, 1]
|ψ> = a|0⟩ + √(1-a²)|1⟩
"""
function density_matrix(a)
    v = [a, sqrt(1 - a^2)]
    return v * v'
end


"""
    get_kraus_operators(gamma, t)
"""
function get_kraus_operators(gamma, t)
    e = exp(-gamma*t)
    k1 = [
        1 0;
        0 sqrt(e)
    ]
    k2 = [
        0 sqrt(1-e)
        0 0
    ]
    return [k1, k2]
end


"""
    amplitude_damping_channel(gamma, t, ρ)
"""
function amplitude_damping_channel(kraus, ρ)
    return sum([k*ρ*k' for k in kraus])
end


"""
    recovery_map(gamma, t, σ, ρ)
"""
function recovery_map(kraus, σ, ρ)
    # Define the evolution map and its adjoint
    map(ρ) = sum([k*ρ*k' for k in kraus])
    map_adj(ρ) = sum([k'*ρ*k for k in kraus])

    inner = matrix_power_pseudo(map(σ), -0.5) * ρ * matrix_power_pseudo(map(σ), -0.5)
    recovered = matrix_power_pseudo(σ, 0.5) * map_adj(inner) * matrix_power_pseudo(σ, 0.5)

    return recovered
end
