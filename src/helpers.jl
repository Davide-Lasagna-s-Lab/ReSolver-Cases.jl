# =============================================================================================== #
# Bounded-domain geometry and canonical cavity liftings                                          #
# =============================================================================================== #

"""Return the underlying global grid used to define a wrapped grid's physical geometry."""
function _global_case_grid(g::AbstractGrid)
    # ---- walk up the parent chain until it stops changing ----
    current = g
    while applicable(Base.parent, current)
        next = parent(current)
        next === current && break
        current = next
    end
    return current
end

"""Return the global physical interval of one bounded storage direction."""
function _bounded_interval(g::AbstractGrid, dim::Int)
    x = vec(points(_global_case_grid(g))[dim])
    return extrema(x)
end

"""Return one bounded coordinate normalized from its physical interval to `[0,1]`."""
function _normalized_bounded_coordinate(g::AbstractGrid, dim::Int)
    x          = vec(points(g)[dim])
    xmin, xmax = _bounded_interval(g, dim)
    return (x .- xmin) ./ (xmax - xmin)
end

"""Return the divergence-free `(U,V)` polynomial shared by the 2D and 3D cavity liftings."""
function _lid_driven_cavity_xy_base(g::AbstractGrid)
    # ---- bounded coordinates normalized to [0,1] ----
    xdim, ydim = inhomogeneous_storage_dims(g)[1:2]
    ξ          = _normalized_bounded_coordinate(g, xdim)
    η          = _normalized_bounded_coordinate(g, ydim)

    # ---- polynomial factors F(ξ), H(η) and their derivatives ----
    F  = @. 16ξ^2 * (1 - ξ)^2
    dF = @. 32ξ * (1 - ξ) * (1 - 2ξ)
    H  = @. η^2 * (η - 1)
    dH = @. 3η^2 - 2η

    # ---- U = F(ξ)H'(η), V = -F'(ξ)H(η); outer products over (x, y) ----
    return F * transpose(dH), -dF * transpose(H)
end

# ============================================================================================== #
# Finite-difference operators                                                                    #
# ============================================================================================== #

"""
    fd_operators(x, w, width) -> (D₁, D₂, D₁⁺, D₂⁺)

FDGrids first- and second-derivative matrices of odd stencil `width` on the points `x`, and their
discrete adjoints with respect to the quadrature weights `w`.
"""
function fd_operators(x::AbstractVector, w::AbstractVector, width::Integer)
    D₁ = FDGrids.DiffMatrix(x, width, 1; eltype=eltype(x))
    D₂ = FDGrids.DiffMatrix(x, width, 2; eltype=eltype(x))

    return D₁, D₂, adjoint(D₁, w), adjoint(D₂, w)
end
