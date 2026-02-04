using Serialization
using StatsBase
using LinearAlgebra

using DecoKiller

const GAMMA = 1.0
const TIMES = range(0.0, 1, 101)  # The number of items should be even
const N_STATES = 16 # How many states generated
const N_QUBITS = 5 # Qubits to create a logic qubit
const BETA = 0.5

const I2 = [1.0+0.0im 0.0; 0.0 1.0]
# Ground and excited states of one qubit
const g0 = [0.0 + 0.0im; 1.0]
const e1 = [1.0 + 0.0im; 0.0]
const SIGMA_X = [0.0+0.0im 1.0; 1.0 0.0+0.0im]
const SIGMA_Y = [0.0+0.0im -1.0im; 1.0im 0.0+0.0im]
const noise_probabilities = normalize([0.8, 0.2])

function collision(ρ, μ)
  return tensor(ρ, μ)
end


function recovery_state(steps=1000)
  noise = sample([SIGMA_X, SIGMA_Y], Weights(noise_probabilities))

  σx_count = σy_count = 0

  ρ = e1 * e1'
  μ = g0 * g0'

  for i in 1:steps
    σ = collision(ρ, μ)

  end


end
