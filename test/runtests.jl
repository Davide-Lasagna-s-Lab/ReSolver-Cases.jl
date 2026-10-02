using Test
import FDGrids
using ReSolverFlowsBase, ReSolverEquations, ReSolverTensorProductGrids, ReSolverCases

include("helpers/analytical.jl")

const FLAGS = FFTW.ESTIMATE

# ---- helpers ----

# full spectral VectorField of `ncomp` zeros carrying `base` in its zero Fourier mode
function base_field(g, base, ncomp)
    u = VectorField(g, FTField, N=ncomp)
    add_base_flow!(u, base)
    return u
end

# largest absolute coefficient of a VectorField
maxabs(u) = maximum(c -> maximum(abs, parent(c)), u.elements)

# the three projected operators, their modes and the shared workspace
function check_operators(ops)
    nl, lin, adj = ops
    @test nl  isa ProjectedEquation{Nonlinear}
    @test lin isa ProjectedEquation{Linearised}
    @test adj isa ProjectedEquation{AdjointDiscrete}
    @test nl.op.work === lin.op.work === adj.op.work
end


@testset "Grids" begin

    @testset "ChannelGrid" begin
        g = ChannelGrid(9, 17, 7; Nt=3, α=1.5, β=2.5, width=5)
        y, x, z, s = points(g)

        # ---- layout ----
        @test g isa AbstractChannelGrid
        @test size(g) == (17, 9, 7, 3)
        @test fft_storage_dims(g) == CHANNEL_FFT_ORDER
        @test inhomogeneous_storage_dims(g) == CHANNEL_INHOMOGENEOUS_DIMS

        # ---- coordinates and quadrature ----
        @test extrema(y) == (-1.0, 1.0)
        @test x[2] ≈ 2π / 1.5 / 9
        @test z[2] ≈ 2π / 2.5 / 7
        @test s[2] ≈ 2π / 3
        @test sum(ReSolverFlowsBase.weights(g)) ≈ 2

        # ---- exact derivatives of a cubic ----
        yv = vec(y)
        @test g.D₁[1] * yv.^3 ≈ 3yv.^2
        @test g.D₂[1] * yv.^3 ≈ 6yv

        # ---- precomputed data and scalar type ----
        h = ChannelGrid(yv, 9, 7, 3, 1.5, 2.5, g.D₁[1], g.D₂[1], g.D₁⁺[1], g.D₂⁺[1], g.ws[1])
        @test size(h) == size(g)
        @test eltype(ChannelGrid(9, 17, 7; T=Float32)) == Float32
    end

    @testset "LidDrivenCavity2DGrid" begin
        g = LidDrivenCavity2DGrid(13; Nt=3, width=5)
        x, y, s = points(g)

        @test g isa AbstractLidDrivenCavity2DGrid
        @test size(g) == (13, 13, 3)
        @test extrema(x) == extrema(y) == (0.0, 1.0)
        @test sum(ReSolverFlowsBase.weights(g)) ≈ 1

        # ---- one discretisation, shared by x and y ----
        @test g.xs[1] === g.xs[2]
        @test g.D₁[1] === g.D₁[2]
    end

    @testset "LidDrivenCavity3DGrid" begin
        g = LidDrivenCavity3DGrid(11; Nt=1, width=5)

        @test g isa AbstractLidDrivenCavity3DGrid
        @test size(g) == (11, 11, 11, 1)
        @test sum(ReSolverFlowsBase.weights(g)) ≈ 1
        @test g.D₂⁺[1] === g.D₂⁺[2] === g.D₂⁺[3]
    end

    @testset "SquareDuctGrid" begin
        g = SquareDuctGrid(13, 7; Nt=3, α=0.5, width=5)
        x, y, z, s = points(g)

        @test g isa AbstractSquareDuctGrid
        @test size(g) == (13, 13, 7, 3)
        @test extrema(x) == (0.0, 1.0)
        @test z[2] ≈ 2π / 0.5 / 7
        @test sum(ReSolverFlowsBase.weights(g)) ≈ 1
    end

    @testset "PipeGrid" begin
        g = PipeGrid(16, 9, 7; Nt=3, α=1.5, R=2)
        r, θ, z, s = points(g)

        @test g isa AbstractPipeGrid
        @test size(g) == (16, 9, 7, 3)
        @test 0 < minimum(r) && maximum(r) == 2
        @test θ[2] ≈ 2π / 9

        # ---- quadrature with the measure r dr: ∫₀² r dr = 2 ----
        @test sum(ReSolverFlowsBase.weights(g)) ≈ 2 rtol=1e-2

        # ---- one-sided stencils at the axis: exact for even and odd polynomials ----
        rv = vec(r)
        @test g.D₁[1] * rv.^2 ≈ 2rv
        @test g.D₁[1] * rv.^3 ≈ 3rv.^2
    end
end


@testset "Grid operators" begin
    include("grids/channel.jl")
    include("grids/lid_driven_cavity_2d.jl")
    include("grids/lid_driven_cavity_3d.jl")
    include("grids/square_duct.jl")
    include("grids/pipe.jl")
end


@testset "Flows" begin

    @testset "PlaneCouetteFlow" begin
        g = ChannelGrid(9, 17, 7; Nt=1, width=5)

        for nlform in (Convective(), Rotational())
            ops = PlaneCouetteFlow(g, 400; nlform, fftw_flags=FLAGS)
            check_operators(ops)

            # ---- laminar Couette U = y is an equilibrium, up to the gradient ∇(|U|²/2) = (0, y, 0)
            #      that the rotational form leaves to the pressure ----
            y = plane_couette_base(g)
            u = base_field(g, (y, nothing, nothing), 3)
            N = ops.nl.op(0, u, similar(u))
            G = nlform isa Rotational ? y : zero(y)
            @test parent(N[2])[:, 1, 1, 1] ≈ G atol=1e-10
            @test maxabs(VectorField(N[1], N[3])) < 1e-10
        end

        # ---- rotation selects the Coriolis force ----
        @test PlaneCouetteFlow(g, 400; Ro=0.1, fftw_flags=FLAGS).nl.op.force isa CoriolisForce
        @test PlaneCouetteFlow(g, 400; fftw_flags=FLAGS).nl.op.force isa NoForce
    end

    @testset "PlanePoiseuilleFlow" begin
        g  = ChannelGrid(9, 17, 7; Nt=1, width=5)
        Re = 180

        # ---- U = 1 - y² is an equilibrium for f = 2/Re ----
        ops = PlanePoiseuilleFlow(g, Re; f=2/Re, fftw_flags=FLAGS)
        check_operators(ops)

        u = base_field(g, (plane_poiseuille_base(g), nothing, nothing), 3)
        @test maxabs(ops.nl.op(0, u, similar(u))) < 1e-10
        @test ops.nl.op.force isa ConstantBodyForce
    end

    @testset "PipeFlow" begin
        g = PipeGrid(16, 7, 5; α=1.5)

        for nlform in (Convective(), Rotational())
            ops = PipeFlow(g, 1000; nlform, fftw_flags=FLAGS)
            check_operators(ops)
            @test ops.nl.op.formulation isa Cylindrical

            # ---- Hagen–Poiseuille is an equilibrium with the default forcing, up to the gradient
            #      ∇(|U|²/2) = (-2r(1 - r²), 0, 0) that the rotational form leaves to the pressure ----
            r = vec(points(g)[1])
            u = base_field(g, (nothing, nothing, hagen_poiseuille_base(g)), 3)
            N = ops.nl.op(0, u, similar(u))
            G = nlform isa Rotational ? @.(-2r * (1 - r^2)) : zero(r)
            @test parent(N[1])[:, 1, 1, 1] ≈ G atol=1e-10
            @test maxabs(VectorField(N[2], N[3])) < 1e-10
        end
    end

    @testset "SquareDuctFlow" begin
        g   = SquareDuctGrid(13, 7; Nt=1, width=5)
        ops = SquareDuctFlow(g, 1000; f=4, fftw_flags=FLAGS)
        check_operators(ops)

        # ---- at rest, the operator is the forcing: f in the mean mode of w only ----
        u = VectorField(g, FTField, N=3)
        N = ops.nl.op(0, u, similar(u))
        @test all(parent(N[3])[:, :, 1, 1] .≈ 4)
        @test maxabs(VectorField(N[1], N[2], N[2])) == 0
    end

    @testset "LidDrivenCavity2DFlow" begin
        g    = LidDrivenCavity2DGrid(13; Nt=1, width=5)
        U, V = lid_driven_cavity_2d_base(g)
        x, y = vec.(points(g)[1:2])

        # ---- lid velocity on the top wall, rest on the others ----
        @test U[:, end] ≈ @. 16x^2 * (1 - x)^2
        @test all(iszero, U[:, 1]) && all(iszero, U[1, :]) && all(iszero, U[end, :])
        @test all(iszero, V[:, [1, end]]) && all(iszero, V[[1, end], :])

        # ---- divergence-free, exactly for these quartics ----
        D = g.D₁[1]
        @test maximum(abs, D * U + transpose(D * transpose(V))) < 1e-10

        for nlform in (Convective(), Rotational())
            check_operators(LidDrivenCavity2DFlow(g, 100; nlform, fftw_flags=FLAGS))
        end
    end

    @testset "LidDrivenCavity3DFlow" begin
        g       = LidDrivenCavity3DGrid(11; Nt=1, width=5)
        U, V, W = lid_driven_cavity_3d_base(g)

        @test size(U) == size(V) == (11, 11, 11)
        @test isnothing(W)

        check_operators(LidDrivenCavity3DFlow(g, 100; fftw_flags=FLAGS))
    end
end
