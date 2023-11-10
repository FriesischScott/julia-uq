include("normalization.jl")
struct GaussianProcess <: UQModel
    gpBase::GPBase
    inputs::Union{Vector{<:UQInput},Vector{Symbol}}
    output::Symbol
    n_sim::Int
    std::InputStandardizationGP
end

function gaussianprocess(
    df::DataFrame,
    inputs::Vector{Symbol},
    output::Symbol,
    kernel::Kernel,
    mean::Mean=MeanZero(),
)
    X = Matrix(df[:, inputs])'
    y = df[:, output]

    X_std = [to_standard(x, minimum(X), maximum(X)) for x in X]

    gp = GP(X_std, y, mean, kernel)
    optimize!(gp)

    gp = GaussianProcess(
        gp, inputs, output, size(X, 2), InputStandardizationGP(minimum(X), maximum(X))
    )

    return gp, df
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

    X_std = [to_standard(x, minimum(X), maximum(X)) for x in X]

    gp = GaussianProcesses.GP(X_std, y, mean, kernel)
    optimize!(gp)

    gp = GaussianProcess(
        gp, inputs, output, size(X, 2), InputStandardizationGP(minimum(X), maximum(X))
    )
    return gp, df
end

### Convenience Functions

function gaussianprocess(
    df::DataFrame, inputs::Symbol, output::Symbol, kernel::Kernel, mean::Mean=MeanZero()
)
    return gaussianprocess(df, [inputs], output, kernel, mean)
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
    if isa(gp.inputs, Vector{Symbol})
        random_names = gp.inputs
    else
        random_names = names(gp.inputs)
    end
    data = df[:, random_names]
    X = Matrix(data)'

    X_std = [to_standard(x, minimum(X), maximum(X)) for x in X]

    df[!, gp.output] = rand(gp.gpBase, X_std)
    return nothing
end

include("plot.jl")