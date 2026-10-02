# Square duct: cross-section x, y bounded on [0, 1] and sharing one discretisation, periodic
# streamwise z, optional periodic time or phase t. Stored as (x, y, z, t).
#
#     g = SquareDuctGrid(49, 63; Nt=1, α=0.5, width=7)
#     x, y, z, t = points(g)

const SQUARE_DUCT_AXES               = (1, 2, 3, 4) # (x, y, z, t) -> storage
const SQUARE_DUCT_FFT_ORDER          = (3, 4)       # z, t
const SQUARE_DUCT_INHOMOGENEOUS_DIMS = (1, 2)       # x, y

"""
    AbstractSquareDuctGrid{T}

Any grid with the square-duct layout.
"""
const AbstractSquareDuctGrid{T} = AbstractGrid{T, 4, SQUARE_DUCT_AXES, SQUARE_DUCT_FFT_ORDER}

"""
    SquareDuctGrid(N, Nz; Nt=1, α=1, dist=FDGrids.GaussLobattoGrid(), width=5, T=Float64)

Square duct stored as `(x, y, z, t)`, with `N` points of distribution `dist` on `[0, 1]` in `x`
and `y`, which share the same points, operators and weights. `Nz`, `Nt` are odd Fourier
resolutions and `α = 2π/Lz`.
"""
const SquareDuctGrid{T, S} = TensorProductGrid{T, S, 4, SQUARE_DUCT_AXES, SQUARE_DUCT_FFT_ORDER}

function SquareDuctGrid(    N::Int,
                           Nz::Int;
                           Nt::Int=1,
                            α::Real=1,
                         dist::FDGrids.AbstractGridDistribution=FDGrids.GaussLobattoGrid(),
                        width::Int=5,
                            T::Type{<:Real}=Float64)

    # ---- one discretisation, shared by x and y ----
    fd = FDGrids.grid(N, 0, 1, dist)
    x  = Vector{T}(fd.xs)
    w  = Vector{T}(fd.ws)

    D₁, D₂, D₁⁺, D₂⁺ = fd_operators(x, w, width)

    # ---- assemble ----
    return TensorProductGrid((x, x), (D₁, D₁), (D₂, D₂), (D₁⁺, D₁⁺), (D₂⁺, D₂⁺), (w, w),
                             (α, 1),
                             (N, N, Nz, Nt),
                             SQUARE_DUCT_AXES,
                             SQUARE_DUCT_FFT_ORDER)
end
