# =============================================================================================== #
# Pressure-driven pipe case                                                                      #
# =============================================================================================== #
#
# The grid layout lives in `grid.jl`. This file provides the Hagen–Poiseuille profile and the
# pressure-driven pipe equations in the cylindrical formulation, with velocity (u_r, u_θ, u_z).
#
#     g = PipeGrid(33, 31, 31; α=1, width=7)
#     nl, lin, adj = PipeFlow(g, 2000; fftw_flags=FFTW.ESTIMATE)
#
# Regularity at the axis is imposed by the basis, not by the grid (see grid.jl).

# =============================================================================================== #
# Hagen–Poiseuille profile                                                                       #
# =============================================================================================== #

"""
    hagen_poiseuille_base(g::AbstractPipeGrid) -> Vector

Return the Hagen–Poiseuille axial profile ``U_z(r) = 1 - (r/R)^2`` at the radial points of `g`,
with unit centreline velocity and zero at the wall ``r = R``.
"""
function hagen_poiseuille_base(g::AbstractPipeGrid)
    r = vec(points(g)[1])
    R = maximum(r)
    return one(eltype(g)) .- (r ./ R).^2
end

# =============================================================================================== #
# Equation constructor                                                                           #
# =============================================================================================== #

@doc raw"""
    PipeFlow(g::AbstractPipeGrid, Re::Real;
             base_flow=(nothing, nothing, hagen_poiseuille_base(g)), f=4/(Re*R^2),
             nlform=Convective(), mode=AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE,
             dealias=true) -> (nl, lin, adj)

Construct the pressure-driven incompressible pipe equations in cylindrical coordinates,

```math
\partial_t \boldsymbol{u} + (\boldsymbol{u}\cdot\nabla)\boldsymbol{u}
= -\nabla p + Re^{-1}\nabla^2\boldsymbol{u} + f\,\boldsymbol{e}_z,
\qquad \nabla\cdot\boldsymbol{u}=0,
```

with ``\boldsymbol{u}=(u_r,u_\theta,u_z)``, radius ``R``, centreline velocity of the reference
profile as velocity scale, and ``Re = U_c R/\nu``. The default forcing ``f = 4/(Re R^2)`` makes the
Hagen–Poiseuille profile ``U_z = 1 - (r/R)^2`` an exact equilibrium.

`base_flow` is added only to the steady zero `(θ, z, t)` Fourier mode; every nonzero component must
have shape `(Nr,)`. The grid and equation constructor do not impose wall values.

# Arguments

- `g`: pipe grid stored as `(r, θ, z, t)`.
- `Re`: Reynolds number ``U_c R/\nu``, multiplying viscosity as ``1/Re``.

# Keyword arguments

- `base_flow`: three-component radial tuple `(U_r, U_θ, U_z)`.
- `f`: nonzero uniform axial forcing; positive values drive flow in ``+z``.
- `nlform`: nonlinearity form, `Convective()` or `Rotational()`.
- `mode`: `AdjointDiscrete()` or `AdjointContinuous()` for the adjoint operator.
- `fftw_flags`: FFTW planner flags forwarded to `ReSolverEquations.construct_equations`.
- `dealias`: whether nonlinear products use padded Fourier resolutions.

# Returns

The projected nonlinear, linearised and adjoint operators `(nl, lin, adj)`, each a
`ReSolverEquations.ProjectedEquation`.
"""
function PipeFlow(         g::AbstractPipeGrid,
                          Re::Real;
                   base_flow=(nothing, nothing, hagen_poiseuille_base(g)),
                           f::Real=4/(Re*maximum(points(g)[1])^2),
                      nlform=Convective(),
                        mode=AdjointDiscrete(),
                  fftw_flags=FFTW.EXHAUSTIVE,
                     dealias::Bool=true)

    # ---- axial pressure-gradient force, on u_z ----
    force = ConstantBodyForce(eltype(g)(f); i=3)

    # ---- projected operators, cylindrical formulation ----
    return construct_equations(g, Re, base_flow, Cylindrical(g);
                               nlform,
                               force,
                               mode,
                               flags=fftw_flags,
                               dealias)
end
