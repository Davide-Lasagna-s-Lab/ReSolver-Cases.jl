# Pipe of radius R: radial r bounded by the wall, periodic azimuthal θ and axial z, optional
# periodic time or phase t. Stored as (r, θ, z, t), θ being the real-to-complex direction.
#
#     g = PipeGrid(33, 31, 31; Nt=1, α=1, R=1, width=7)
#     r, θ, z, t = points(g)
#
# Radial points are the positive half of a Chebyshev–Lobatto grid: the axis is excluded, the wall
# included, and the weights include the measure r dr. Near the axis the derivative stencils are
# one-sided, built from interior points only, as in Openpipeflow (Willis 2017): the same matrices
# serve every azimuthal wavenumber and component, whatever their parity across the axis.
# Regularity at the axis is imposed by the basis, whose modes are built per wavenumber.

const PIPE_AXES               = (1, 2, 3, 4) # (r, θ, z, t) -> storage
const PIPE_FFT_ORDER          = (2, 3, 4)    # θ, z, t
const PIPE_INHOMOGENEOUS_DIMS = (1,)         # r

"""
    AbstractPipeGrid{T}

Any grid with the pipe layout.
"""
const AbstractPipeGrid{T} = AbstractGrid{T, 4, PIPE_AXES, PIPE_FFT_ORDER}

"""
    PipeGrid(Nr, Nθ, Nz; Nt=1, α=1, R=1, width=7, T=Float64)

Pipe of radius `R` stored as `(r, θ, z, t)`, with `Nr` radial points on `(0, R]` and FDGrids
stencils of odd `width` (7 or 9 recommended). `Nθ`, `Nz`, `Nt` are odd Fourier resolutions; the
azimuthal period is `2π` and `α = 2π/Lz`.
"""
const PipeGrid{T, S} = TensorProductGrid{T, S, 4, PIPE_AXES, PIPE_FFT_ORDER}

function PipeGrid(   Nr::Int,
                     Nθ::Int,
                     Nz::Int;
                     Nt::Int=1,
                      α::Real=1,
                      R::Real=1,
                  width::Int=7,
                      T::Type{<:Real}=Float64)

    # ---- radial points, weights with r dr, and operators ----
    fd = FDGrids.grid(Nr, T(R), FDGrids.HalfChebyshevGrid())
    r  = Vector{T}(fd.xs)
    w  = Vector{T}(fd.ws)

    D₁, D₂, D₁⁺, D₂⁺ = fd_operators(r, w, width)

    # ---- assemble ----
    return TensorProductGrid((r,), (D₁,), (D₂,), (D₁⁺,), (D₂⁺,), (w,),
                             (1, α, 1),
                             (Nr, Nθ, Nz, Nt),
                             PIPE_AXES,
                             PIPE_FFT_ORDER)
end
