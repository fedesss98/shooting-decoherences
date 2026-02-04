module DecoKiller

export 
PetzMaps,
get_dephasing_operators, get_amplitudedamping_operators, get_bitflip_operators, 
# initial states
density_matrix, codespace_state, thermal_state, random_state,
apply_channel, recovery_map,
# metrics
fidelity, overlap,

UnitaryDilation,
PetzCollisionModel, apply_petz_collision, extract_kraus_operators


include("PetzMaps.jl")
include("UnitaryDilation/UnitaryDilation.jl")

include("utils.jl")

using .PetzMaps
using .UnitaryDilation

end
