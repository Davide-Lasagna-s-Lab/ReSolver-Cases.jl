# Channel: wall-normal y bounded on [-1, 1], periodic streamwise x and spanwise z, optional
# periodic time or phase t. Stored as (y, x, z, t), x being the real-to-complex direction.
#
#     g = ChannelGrid(63, 65, 63; Nt=1, α=0.5, β=1, width=7)
#     y, x, z, t = points(g)

const CHANNEL_AXES               = (2, 1, 3, 4) # (x, y, z, t) -> storage
const CHANNEL_FFT_ORDER          = (2, 3, 4)    # x, z, t
const CHANNEL_INHOMOGENEOUS_DIMS = (1,)         # y

"""
    AbstractChannelGrid{T}

Any grid with the channel layout, including device and decomposed wrappers.
"""
const AbstractChannelGrid{T} = AbstractGrid{T, 4, CHANNEL_AXES, CHANNEL_FFT_ORDER}

"""
    ChannelGrid(Nx, Ny, Nz; Nt=1, α=1, β=1, dist=FDGrids.GaussLobattoGrid(), width=5, T=Float64)
    ChannelGrid(y, Nx, Nz, Nt, α, β, Dy, Dy2, Dya, Dy2a, wy, T=Float64)

Channel grid stored as `(y, x, z, t)`, with `Ny` wall-normal points of distribution `dist` on
`[-1, 1]` and FDGrids stencils of odd `width`, or with given wall-normal points, derivative
matrices, discrete adjoints and weights. `Nx`, `Nz`, `Nt` are odd Fourier resolutions;
`α = 2π/Lx`, `β = 2π/Lz`, and the phase `t ∈ [0, 2π)` has scale one.
"""
const ChannelGrid{T, S} = TensorProductGrid{T, S, 4, CHANNEL_AXES, CHANNEL_FFT_ORDER}

function ChannelGrid(   Nx::Int,
                        Ny::Int,
                        Nz::Int;
                        Nt::Int=1,
                         α::Real=1,
                         β::Real=1,
                      dist::FDGrids.AbstractGridDistribution=FDGrids.GaussLobattoGrid(),
                     width::Int=5,
                         T::Type{<:Real}=Float64)

    # ---- wall-normal points, weights and operators ----
    fd = FDGrids.grid(Ny, -1, 1, dist)
    y  = Vector{T}(fd.xs)
    w  = Vector{T}(fd.ws)

    Dy, Dy2, Dya, Dy2a = fd_operators(y, w, width)

    # ---- assemble ----
    return ChannelGrid(y, Nx, Nz, Nt, α, β, Dy, Dy2, Dya, Dy2a, w, T)
end

function ChannelGrid(   y::AbstractVector,
                       Nx::Int,
                       Nz::Int,
                       Nt::Int,
                        α::Real,
                        β::Real,
                       Dy::AbstractMatrix,
                      Dy2::AbstractMatrix,
                      Dya::AbstractMatrix,
                     Dy2a::AbstractMatrix,
                       wy::AbstractVector,
                         ::Type{T}=Float64) where {T<:Real}

    # ---- grid of the given data, converted to T ----
    g = TensorProductGrid((y,), (Dy,), (Dy2,), (Dya,), (Dy2a,), (wy,),
                          (α, β, 1),
                          (length(y), Nx, Nz, Nt),
                          CHANNEL_AXES,
                          CHANNEL_FFT_ORDER)

    return convert(T, g)
end
