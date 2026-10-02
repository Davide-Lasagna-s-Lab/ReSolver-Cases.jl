@testset verbose=true "Pipe grid                                                   " begin
    g = PipeGrid(24, 9, 19; Nt=3, α=1.5, width=7)

    # a regular field with azimuthal wavenumber 1 (odd in r across the axis)
    u(r, θ, z, t)  = r * (1 - r^2) * cos(θ) * periodic_profile(1.5z) * periodic_profile(t)
    ur(r, θ, z, t) = (1 - 3r^2) * cos(θ) * periodic_profile(1.5z) * periodic_profile(t)
    uθ(r, θ, z, t) = -r * (1 - r^2) * sin(θ) * periodic_profile(1.5z) * periodic_profile(t)
    uz(r, θ, z, t) = r * (1 - r^2) * cos(θ) * 1.5 * periodic_profile_d1(1.5z) * periodic_profile(t)

    @testset verbose=true "Analytical derivatives                                      " begin
        û = FFT(Field(g, u))
        for (derivative!, exact) in ((ddr!, ur), (ddθ!, uθ), (ddz!, uz))
            @test derivative!(FTField(g), û) ≈ FFT(Field(g, exact)) atol=3e-7 rtol=3e-7
        end
    end

    @testset verbose=true "Quadrature-weighted discrete adjoints                       " begin
        v(r, θ, z, t) = (1 + r^2) * periodic_profile(θ + 0.2) * periodic_profile(1.5z + 0.3) *
                        periodic_profile(t + 0.4)
        û, v̂ = FFT(Field(g, u)), FFT(Field(g, v))

        for derivative! in (ddr!, ddθ!, ddz!, dds!)
            Du  = derivative!(FTField(g), û)
            D⁺v = derivative!(FTField(g), v̂, DiscreteAdjoint())
            @test dot(Du, v̂) ≈ dot(û, D⁺v) atol=5e-12 rtol=5e-12
        end
    end
end
