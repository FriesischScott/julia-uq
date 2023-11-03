struct GaussianProcess <: UQModel
    gpBase::GPBase
    inputs::Vector{<:UQInput}
    output::Symbol
    n_sim::Int
end

function gaussianprocess(
    df::DataFrame,
    inputs::Vector{<:UQInput},
    output::Symbol,
    kernel::Kernel,
    mean::Mean=MeanZero(),
)
    random_inputs = filter(i -> isa(i, RandomUQInput), inputs)
    random_names = names(random_inputs)
    X = Matrix(df[:, random_names])'
    y = df[:, output]

    gp = GaussianProcesses.GP(X, y, mean, kernel)
    optimize!(gp)

    GP = GaussianProcess(gp, inputs, output, size(X, 2))

    return GP, df
end

function gaussianprocess(
    inputs::Vector{<:UQInput},
    model::Vector{<:UQModel},
    output::Symbol,
    sim::AbstractMonteCarlo,
    kernel::Kernel,
    mean::Mean=MeanZero(),
)
    df = sample(inputs, sim)
    evaluate!(model, df)

    return gaussianprocess(df, inputs, output, kernel, mean)
end

function gaussianprocess(
    inputs::UQInput,
    model::Vector{<:UQModel},
    output::Symbol,
    sim::AbstractMonteCarlo,
    kernel::Kernel,
    mean::Mean=MeanZero(),
)
    return gaussianprocess([inputs], model, output, sim, kernel, mean)
end

function gaussianprocess(
    inputs::Vector{<:UQInput},
    model::UQModel,
    output::Symbol,
    sim::AbstractMonteCarlo,
    kernel::Kernel,
    mean::Mean=MeanZero(),
)
    return gaussianprocess(inputs, [model], output, sim, kernel, mean)
end

function gaussianprocess(
    inputs::UQInput,
    model::UQModel,
    output::Symbol,
    sim::AbstractMonteCarlo,
    kernel::Kernel,
    mean::Mean=MeanZero(),
)
    return gaussianprocess([inputs], [model], output, sim, kernel, mean)
end

function evaluate!(gp::GaussianProcess, df::DataFrame)
    data = df[:, names(gp.inputs)]
    X = Matrix(data)'
    df[!, gp.output] = rand(gp.gpBase, X)
    return nothing
end