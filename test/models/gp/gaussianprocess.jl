
@testset "GaussianProcess" begin
    random_names = [:x1, :x2, :x3]
    x = RandomVariable.([Uniform(-2, 0), Normal(-1, 0.5), Uniform(0, 1)], random_names)

    model1 = Model(df -> begin
        return df.x1 .+ df.x2 .* df.x3
    end, :y1)

    model2 = Model(df -> begin
        return df.y1
    end, :y)

    model = [model1, model2]

    kernel = SE([0.0, 0.0, 0.0], 0.0)
    m = MeanZero()

    sim = MonteCarlo(10)

    df = sample(x, sim)
    evaluate!(model, df)

    gp, samples = gaussianprocess(df, random_names, :y, kernel, m)

    @test gp.inputs == random_names
    @test gp.output == :y
    @test gp.n_sim == sim.n
    @test isa(gp.gpBase, GPE)

    gp, samples = gaussianprocess(df, x, :y, kernel, m)

    @test gp.inputs == x
    @test gp.output == :y
    @test gp.n_sim == sim.n
    @test isa(gp.gpBase, GPE)

    @testset "Convenience Functions" begin
        x1 = RandomVariable(Uniform(-2, 0), :x1)
        x2 = RandomVariable(Uniform(-2, 0), :x2)

        model_a = Model(df -> begin
            return df.x1 .^ 2
        end, :ya)

        model_b = Model(df -> begin
            return df.ya .* 2
        end, :yb)

        sim = MonteCarlo(10)

        kernel_1 = SE(0.0, 0.0)
        kernel_2 = SE([0.0, 0.0], 0.0)
        m = MeanZero()

        df = sample(x1, sim)
        evaluate!(model_a, df)

        gp_1, _ = gaussianprocess(df, :x1, :ya, kernel, m)

        gp_11, _ = gaussianprocess(x1, model_a, :ya, sim, kernel, m)

        gp_12, _ = gaussianprocess(x1, [model_a, model_b], :yb, sim, kernel, m)

        gp_21, _ = gaussianprocess([x1, x2], model_a, :ya, sim, kernel, m)

        @test isa(gp_1, GaussianProcess)
        @test isa(gp_11, GaussianProcess)
        @test isa(gp_12, GaussianProcess)
        @test isa(gp_21, GaussianProcess)
    end

    @testset "evaluate" begin
        gp, samples = gaussianprocess(x, model, :y, sim, kernel)

        data = copy(samples)
        evaluate!(gp, data)

        @test sum((samples.y .- data.y) .^ 2) ≈ 0.0 atol = 0.01

        df = sample(x, sim)
        evaluate!(model, df)

        gp, samples = gaussianprocess(df, names(x), :y, kernel, m)

        data = copy(samples)
        evaluate!(gp, data)
        @test sum((samples.y .- data.y) .^ 2) ≈ 0.0 atol = 0.01
    end
end
