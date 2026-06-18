import Pkg

Pkg.activate(joinpath(@__DIR__, "julia"))

Pkg.instantiate()
Pkg.add("IJulia")

using IJulia

kernel_name = "DecoKiller"

IJulia.installkernel(
    kernel_name,
    "--project=$(dirname(Base.active_project()))"
)

println("Done. In VS Code, select the '$kernel_name' kernel for notebooks.")