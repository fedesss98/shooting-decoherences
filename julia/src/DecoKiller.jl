module DecoKiller

include("PetzMaps.jl")
include("UnitaryDilation/UnitaryDilation.jl")
include("BB84Circuit.jl")

export fidelity, overlap, encode_bit, decode_bit, partial_trace

include("utils.jl")

using .PetzMaps
using .UnitaryDilation
using .BB84Circuit

end
