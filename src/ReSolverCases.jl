module ReSolverCases

# Concrete grids and flow cases: channel (plane Couette, plane Poiseuille), two- and
# three-dimensional lid-driven cavities, square duct and pipe. Each case folder holds its grid and
# its equation constructors.

import FDGrids

using ReSolverFlowsBase: AbstractGrid, FFTW, inhomogeneous_storage_dims, points

using ReSolverTensorProductGrids: TensorProductGrid

using ReSolverEquations: AdjointDiscrete, Cartesian, ConstantBodyForce, Convective, CoriolisForce,
                         Cylindrical, NoForce, construct_equations

# ---- channel ----
export CHANNEL_AXES, CHANNEL_FFT_ORDER, CHANNEL_INHOMOGENEOUS_DIMS
export AbstractChannelGrid, ChannelGrid
export plane_couette_base, plane_poiseuille_base, PlaneCouetteFlow, PlanePoiseuilleFlow

# ---- lid-driven cavities ----
export LID_DRIVEN_CAVITY_2D_AXES, LID_DRIVEN_CAVITY_2D_FFT_ORDER, LID_DRIVEN_CAVITY_2D_INHOMOGENEOUS_DIMS
export AbstractLidDrivenCavity2DGrid, LidDrivenCavity2DGrid
export lid_driven_cavity_2d_base, LidDrivenCavity2DFlow

export LID_DRIVEN_CAVITY_3D_AXES, LID_DRIVEN_CAVITY_3D_FFT_ORDER, LID_DRIVEN_CAVITY_3D_INHOMOGENEOUS_DIMS
export AbstractLidDrivenCavity3DGrid, LidDrivenCavity3DGrid
export lid_driven_cavity_3d_base, LidDrivenCavity3DFlow

# ---- square duct ----
export SQUARE_DUCT_AXES, SQUARE_DUCT_FFT_ORDER, SQUARE_DUCT_INHOMOGENEOUS_DIMS
export AbstractSquareDuctGrid, SquareDuctGrid, SquareDuctFlow

# ---- pipe ----
export PIPE_AXES, PIPE_FFT_ORDER, PIPE_INHOMOGENEOUS_DIMS
export AbstractPipeGrid, PipeGrid, hagen_poiseuille_base, PipeFlow

include("helpers.jl")

include("channel/grid.jl")
include("channel/case.jl")

include("lid_driven_cavity_2d/grid.jl")
include("lid_driven_cavity_2d/case.jl")

include("lid_driven_cavity_3d/grid.jl")
include("lid_driven_cavity_3d/case.jl")

include("square_duct/grid.jl")
include("square_duct/case.jl")

include("pipe/grid.jl")
include("pipe/case.jl")

end
