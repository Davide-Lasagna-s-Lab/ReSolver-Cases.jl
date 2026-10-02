# Square two-dimensional cavity: x and y bounded and sharing one discretisation, optional
# periodic time or phase t. Stored as (x, y, t).
#
#     g = LidDrivenCavity2DGrid(65; Nt=1, width=7)
#     x, y, t = points(g)

const LID_DRIVEN_CAVITY_2D_AXES               = (1, 2, nothing, 3) # (x, y, z, t) -> storage
const LID_DRIVEN_CAVITY_2D_FFT_ORDER          = (3,)               # t
const LID_DRIVEN_CAVITY_2D_INHOMOGENEOUS_DIMS = (1, 2)             # x, y

"""
    AbstractLidDrivenCavity2DGrid{T}

Any grid with the two-dimensional cavity layout.
"""
const AbstractLidDrivenCavity2DGrid{T} =
    AbstractGrid{T, 3, LID_DRIVEN_CAVITY_2D_AXES, LID_DRIVEN_CAVITY_2D_FFT_ORDER}

"""
    LidDrivenCavity2DGrid(N; Nt=1, lim=(0, 1), dist=FDGrids.UniformGrid(), width=5, T=Float64)

Square cavity stored as `(x, y, t)`, with `N` points of distribution `dist` on `lim` in both `x`
and `y`, which share the same points, operators and weights. `Nt` is an odd temporal resolution.
"""
const LidDrivenCavity2DGrid{T, S} =
    TensorProductGrid{T, S, 3, LID_DRIVEN_CAVITY_2D_AXES, LID_DRIVEN_CAVITY_2D_FFT_ORDER}

function LidDrivenCavity2DGrid(    N::Int;
                                  Nt::Int=1,
                                 lim::NTuple{2, <:Real}=(0, 1),
                                dist::FDGrids.AbstractGridDistribution=FDGrids.UniformGrid(),
                               width::Int=5,
                                   T::Type{<:Real}=Float64)

    # ---- one discretisation, shared by x and y ----
    fd = FDGrids.grid(N, lim[1], lim[2], dist)
    x  = Vector{T}(fd.xs)
    w  = Vector{T}(fd.ws)

    D₁, D₂, D₁⁺, D₂⁺ = fd_operators(x, w, width)

    # ---- assemble ----
    return TensorProductGrid((x, x), (D₁, D₁), (D₂, D₂), (D₁⁺, D₁⁺), (D₂⁺, D₂⁺), (w, w),
                             (1,),
                             (N, N, Nt),
                             LID_DRIVEN_CAVITY_2D_AXES,
                             LID_DRIVEN_CAVITY_2D_FFT_ORDER)
end
