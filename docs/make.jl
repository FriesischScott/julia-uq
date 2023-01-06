using Documenter, UncertaintyQuantification

DocMeta.setdocmeta!(
    UncertaintyQuantification,
    :DocTestSetup,
    :(using UncertaintyQuantification, Random; Random.seed!(8128));
    recursive=true,
)

makedocs(;
    modules=[UncertaintyQuantification],
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true"),
    sitename="UncertaintyQuantification.jl",
    authors="Jasper Behrensdorf and Ander Gray",
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "Uncertainty Propagation" => "manual/uncertainty.md",
            "Sensitivity Analysis" => "manual/sensitivity.md",
            "Reliability Analysis" => "manual/reliability.md",
        ],
        "API" => [
            "ExternalModel" => "externalmodel.md",
            "GaussianCopula" => "api/gaussian.md",
            "Inputs" => "api/inputs.md",
            "JointDistribution" => "api/jointdistribution.md",
            "Model" => "api/model.md",
            "PolynomialChaosExpansion" => "polynomialchaosexpansion.md",
            "Parameter" => "api/parameter.md",
            "RandomVariable" => "api/randomvariable.md",
            "ResponseSurface" => "api/responsesurface.md",
            "PolyharmonicSpline" => "api/polyharmonicspline.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/FriesischScott/UncertaintyQuantification.jl", push_preview=true
)
