struct GaussianProcess <: UQModel
    gpBase::GPBase
    inputs::Vector{<:UQInput}
    output::Symbol
    model::Vector{<:UQModel}
    sim::AbstractMonteCarlo
end

function gaussianprocess(
    inputs::Vector{<:UQInput},
    model::Vector{<:UQModel},
    output::Symbol,
    sim::AbstractMonteCarlo,
    kernel::Kernel,
    mean::Mean=ZeroMean(),
)
    samples = sample(inputs, sim)
    evaluate!(model, samples)

    random_inputs = filter(i -> isa(i, RandomUQInput), inputs)
    random_names = names(random_inputs)
    X = Matrix(samples[:, random_names])'
    y = samples[:, output]
    gp = GP(X, y, mean, kernel)
    optimize!(gp)

    return GaussianProcess(gp, inputs, output, model, sim)
end

function gaussianprocess(
    inputs::UQInput,
    model::Vector{<:UQModel},
    output::Symbol,
    sim::AbstractMonteCarlo,
    kernel::Kernel,
    mean::Mean=ZeroMean(),
)
    return gaussianprocess([inputs], model, output, sim, kernel, mean)
end

function gaussianprocess(
    inputs::Vector{<:UQInput},
    model::UQModel,
    output::Symbol,
    sim::AbstractMonteCarlo,
    kernel::Kernel,
    mean::Mean=ZeroMean(),
)
    return gaussianprocess(inputs, [model], output, sim, kernel, mean)
end

function gaussianprocess(
    inputs::UQInput,
    model::UQModel,
    output::Symbol,
    sim::AbstractMonteCarlo,
    kernel::Kernel,
    mean::Mean=ZeroMean(),
)
    return gaussianprocess([inputs], [model], output, sim, kernel, mean)
end

function evaluate!(gp::GaussianProcess, df::DataFrame)
    data = df[:, names(gp.inputs)]
    X = Matrix(data)'
    df[!, gp.output] = rand(gp.gpBase, X)
    return nothing
end