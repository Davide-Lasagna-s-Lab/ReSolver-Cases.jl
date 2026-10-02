# ReSolverCases.jl

Concrete grids and flow cases for the ReSolver packages, each in its own folder with its grid and
its equation constructors:

| Case | Grid | Flows |
|---|---|---|
| channel | `ChannelGrid` | `PlaneCouetteFlow`, `PlanePoiseuilleFlow` |
| two-dimensional cavity | `LidDrivenCavity2DGrid` | `LidDrivenCavity2DFlow` |
| three-dimensional cavity | `LidDrivenCavity3DGrid` | `LidDrivenCavity3DFlow` |
| square duct | `SquareDuctGrid` | `SquareDuctFlow` |
| pipe | `PipeGrid` | `PipeFlow` |

```julia
using ReSolverFlowsBase, ReSolverEquations, ReSolverCases

g = ChannelGrid(33, 65, 33; Nt=1, α=1, β=2, width=7)

nl, lin, adj = PlaneCouetteFlow(g, 400; nlform=Rotational())

nl(out, a)                # nonlinear operator on modal coefficients
linearise_about!(lin, a)  # linearisation point, shared by lin and adj
lin(out, b)               # linearised operator
adj(out, b)               # adjoint operator
```

Grids are [ReSolverTensorProductGrids](https://github.com/Davide-Lasagna-s-Lab/ReSolver-TensorProductGrids.jl)
built with [FDGrids](https://github.com/Davide-Lasagna-s-Lab/FDGrids.jl) operators; the equations
come from [ReSolverEquations](https://github.com/Davide-Lasagna-s-Lab/ReSolver-Equations.jl).
Every flow takes `nlform` (`Convective()` or `Rotational()`) and `mode` (`AdjointDiscrete()` or
`AdjointContinuous()`).
