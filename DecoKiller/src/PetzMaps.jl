module PetzMaps

using LinearAlgebra
using Random

export
    get_depolarizing_operators,
    get_amplitudedamping_operators,
    get_bitflip_operators,
    get_phasedamping_operators,
    get_random_operators,
    get_pin_operators,
    correlate_kraus_operators,
    tensor_power,
    normalize_kraus_operators,
    # initial states
    codespace_state, codespace_dm, input_state, thermal_state, random_state,
    thermal_state_hopping,
    single_excitation_state, single_excitation_dm,
    apply_channel, apply_extended_channel, recovery_map, rand_state_with_spectrum, expand_kraus_operators,
    # metrics
    fidelity, overlap

include("utils.jl")
include("operators.jl")
include("metrics.jl")
include("quantum_states.jl")

end
