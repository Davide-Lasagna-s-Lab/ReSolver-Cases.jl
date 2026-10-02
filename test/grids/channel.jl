@testset verbose=true "Channel grid                                                " begin
    Ny, Nx, Nz, Nt = 17, 19, 19, 19
    α, β, ylim = 1.25, 0.75, (-1, 1)
    g = ChannelGrid(Nx, Ny, Nz; Nt, α, β, dist=FDGrids.GaussLobattoGrid(), width=7)

    u(y, x, z, t) = bounded_profile(y, ylim) * periodic_profile(α * x) *
                     periodic_profile(β * z) * periodic_profile(t)
    ux(y, x, z, t) = bounded_profile(y, ylim) * α * periodic_profile_d1(α * x) *
                      periodic_profile(β * z) * periodic_profile(t)
    uy(y, x, z, t) = bounded_profile_d1(y, ylim) * periodic_profile(α * x) *
                      periodic_profile(β * z) * periodic_profile(t)
    uz(y, x, z, t) = bounded_profile(y, ylim) * periodic_profile(α * x) *
                      β * periodic_profile_d1(β * z) * periodic_profile(t)
    ut(y, x, z, t) = bounded_profile(y, ylim) * periodic_profile(α * x) *
                      periodic_profile(β * z) * periodic_profile_d1(t)
    Δu(y, x, z, t) = (bounded_profile_d2(y, ylim) * periodic_profile(α * x) *
        periodic_profile(β * z) + bounded_profile(y, ylim) *
        (α^2 * periodic_profile_d2(α * x) * periodic_profile(β * z) +
         β^2 * periodic_profile(α * x) * periodic_profile_d2(β * z))) *
        periodic_profile(t)

    @testset verbose=true "Analytical derivatives and Laplacian                        " begin
        û = FFT(Field(g, u))
        @test ddy!(FTField(g), û) ≈ FFT(Field(g, uy)) atol=3e-11 rtol=3e-11
        for (derivative!, exact) in ((ddx!, ux), (ddz!, uz), (ddt!, ut))
            @test derivative!(FTField(g), û) ≈ FFT(Field(g, exact)) atol=3e-7 rtol=3e-7
        end
        @test laplacian!(FTField(g), û) ≈ FFT(Field(g, Δu)) atol=3e-7 rtol=3e-7
    end

    @testset verbose=true "Analytical norms and homogeneous shifts                     " begin
        û = FFT(Field(g, u))
        exact_norm2 = Float64(bounded_profile_norm2(ylim) * PERIODIC_PROFILE_NORM2^3)
        velocity = VectorField(Field(g, u), Field(g, (y, x, z, t) -> 2u(y, x, z, t)),
                               Field(g, (y, x, z, t) -> 3u(y, x, z, t)))
        @test norm(û)^2 ≈ exact_norm2 rtol=3e-12
        @test norm(FFT(velocity))^2 ≈ 14exact_norm2 rtol=3e-12

        sx, sz, st = 0.37, 1.11, 0.23
        shifted(y, x, z, t) = bounded_profile(y, ylim) * periodic_profile(α * (x + sx)) *
                               periodic_profile(β * (z + sz)) * periodic_profile(t + st)
        @test shift!(copy(û), (sx, sz, st)) ≈ FFT(Field(g, shifted)) atol=3e-7 rtol=3e-7
    end

    @testset verbose=true "Quadrature-weighted discrete adjoints                       " begin
        v(y, x, z, t) = dual_bounded_profile(y, ylim) * periodic_profile(α * x + 0.2) *
                         periodic_profile(β * z + 0.3) * periodic_profile(t + 0.4)
        û, v̂ = FFT(Field(g, u)), FFT(Field(g, v))

        for derivative! in (ddx!, ddy!, ddz!, ddt!)
            Du = derivative!(FTField(g), û)
            D⁺v = derivative!(FTField(g), v̂, DiscreteAdjoint())
            @test dot(Du, v̂) ≈ dot(û, D⁺v) atol=5e-12 rtol=5e-12
        end
        Δû = laplacian!(FTField(g), û)
        Δ⁺v = laplacian!(FTField(g), v̂, DiscreteAdjoint())
        @test dot(Δû, v̂) ≈ dot(û, Δ⁺v) atol=5e-10 rtol=5e-10
    end
end
