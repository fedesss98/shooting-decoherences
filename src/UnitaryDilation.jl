module UnitaryDilation

using LinearAlgebra

# Export functions you want users to access
export tensor, qubit_basis, ancilla_basis, complete_basis_qr, build_input_basis, construct_unitary, kraus_operators_recovery1, isometry, chop!

# Include submodules
include("quantum_states.jl")
include("basis.jl")
include("utils.jl")
include("operators.jl")

end # module
