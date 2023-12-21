@testset "Distribution Parameters" begin
    # Beta
    d = Beta(0.5, 0.5)
    μ = mean(d)
    σ = std(d)
    @test [distribution_parameters(μ, σ, Beta)...] ≈ [0.5, 0.5]

    # Gamma
    d = Gamma(3.0, 2.0)
    μ = mean(d)
    σ = std(d)
    @test [distribution_parameters(μ, σ, Gamma)...] ≈ [3.0, 2.0]

    # Gumbel
    d = Gumbel(1.0, 2.0)
    μ = mean(d)
    σ = std(d)
    @test [distribution_parameters(μ, σ, Gumbel)...] ≈ [1.0, 2.0]

    # Logistic
    d = Logistic(5.0, 2.0)
    μ = mean(d)
    σ = std(d)
    @test [distribution_parameters(μ, σ, Logistic)...] ≈ [5.0, 2.0]

    # LogNormal
    d = LogNormal(10.0, 5.0)
    μ = mean(d)
    σ = std(d)
    @test [distribution_parameters(μ, σ, LogNormal)...] ≈ [10.0, 5.0]
end
