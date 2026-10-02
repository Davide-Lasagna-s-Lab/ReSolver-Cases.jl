# Cubic three-dimensional cavity: x, y and z bounded and sharing one discretisation, optional
# periodic time phase s ∈ [0, 2π). Stored as (x, y, z, s).
#
#     g = LidDrivenCavity3DGrid(49; Nt=1, width=7)
#     x, y, z, s = points(g)

const LID_DRIVEN_CAVITY_3D_AXES               = (1, 2, 3, 4) # (x, y, z, s) -> storage
const LID_DRIVEN_CAVITY_3D_FFT_ORDER          = (4,)         # s
const LID_DRIVEN_CAVITY_3D_INHOMOGENEOUS_DIMS = (1, 2, 3)    # x, y, z

"""
    AbstractLidDrivenCavity3DGrid{T}

Any grid with the three-dimensional cavity layout.
"""
const AbstractLidDrivenCavity3DGrid{T} =
    AbstractGrid{T, 4, LID_DRIVEN_CAVITY_3D_AXES, LID_DRIVEN_CAVITY_3D_FFT_ORDER}

"""
    LidDrivenCavity3DGrid(N; Nt=1, lim=(0, 1), dist=FDGrids.UniformGrid(), width=5, T=Float64)

Cubic cavity stored as `(x, y, z, s)`, with `N` points of distribution `dist` on `lim` in `x`, `y`
and `z`, which share the same points, operators and weights. `Nt` is an odd temporal resolution.
"""
const LidDrivenCavity3DGrid{T, S} =
    TensorProductGrid{T, S, 4, LID_DRIVEN_CAVITY_3D_AXES, LID_DRIVEN_CAVITY_3D_FFT_ORDER}

function LidDrivenCavity3DGrid(    N::Int;
                                  Nt::Int=1,
                                 lim::NTuple{2, <:Real}=(0, 1),
                                dist::FDGrids.AbstractGridDistribution=FDGrids.UniformGrid(),
                               width::Int=5,
                                   T::Type{<:Real}=Float64)

    # ---- one discretisation, shared by x, y and z ----
    fd = FDGrids.grid(N, lim[1], lim[2], dist)
    x  = Vector{T}(fd.xs)
    w  = Vector{T}(fd.ws)

    D₁, D₂, D₁⁺, D₂⁺ = fd_operators(x, w, width)

    # ---- assemble ----
    return TensorProductGrid((x, x, x), (D₁, D₁, D₁), (D₂, D₂, D₂),
                             (D₁⁺, D₁⁺, D₁⁺), (D₂⁺, D₂⁺, D₂⁺), (w, w, w),
                             (),
                             (N, N, N, Nt),
                             LID_DRIVEN_CAVITY_3D_AXES,
                             LID_DRIVEN_CAVITY_3D_FFT_ORDER)
end
