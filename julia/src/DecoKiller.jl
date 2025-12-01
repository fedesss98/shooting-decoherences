module DecoKiller

include("PetzMaps.jl")
include("UnitaryDilation.jl")
include("BB84Circuit.jl")

export fidelity, encode_bit, decode_bit

using .PetzMaps
using .UnitaryDilation
using .BB84Circuit

end
