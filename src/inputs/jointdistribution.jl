Correlateable = Union{RandomVariable, ProbabilityBox}

struct JointDistribution <: RandomUQInput
    marginals::Vector{<:UQInput}
    copula::Copula

    function JointDistribution(marginals::Vector{<:UQInput}, copula::Copula)
        length(marginals) == dimensions(copula) ||
            error("Dimension mismatch between copula and marginals")

        all(isa.(marginals, Correlateable)) || error("Only correlateable inputs can be passed to a joint distribution. Correlateable inputs are $(get_union_types(Correlateable))")
        return new(marginals, copula)
    end
end

function sample(jd::JointDistribution, n::Integer=1)
    u = sample(jd.copula, n)

    samples = DataFrame()

    for (i, rv) in enumerate(jd.marginals)
        samples[!, rv.name] = quantile.(Ref(rv), u[:, i])
    end

    return samples
end

function to_physical_space!(jd::JointDistribution, x::DataFrame)
    correlated_cdf = to_copula_space(jd.copula, Matrix{Float64}(x[:, names(jd)]))
    for (i, rv) in enumerate(jd.marginals)
        x[!, rv.name] = quantile.(Ref(rv), correlated_cdf[:, i])
    end
    return nothing
end

function to_standard_normal_space!(jd::JointDistribution, x::DataFrame)
    for rv in jd.marginals
        if isa(rv, RandomVariable)
            x[!, rv.name] = cdf.(rv.dist, x[:, rv.name])
        elseif isa(rv, ProbabilityBox)
            x[!, rv.name] = reverse_quantile.(rv, x[:, rv.name])
        end
    end
    uncorrelated_stdnorm = to_standard_normal_space(
        jd.copula, Matrix{Float64}(x[:, names(jd)])
    )
    for (i, rv) in enumerate(jd.marginals)
        x[!, rv.name] = uncorrelated_stdnorm[:, i]
    end
    return nothing
end

function map_to_precise(x::AbstractVector{<:Real}, jd::JointDistribution)
    
end

marginals(jd::JointDistribution) = jd.marginals
marginals(x::T) where T <:UQInput = [x]

names(jd::JointDistribution) = vec(map(x -> x.name, jd.marginals))

mean(jd::JointDistribution) = mean.(jd.marginals)

dimensions(jd::JointDistribution) = dimensions(jd.copula)

bounds(jd::JointDistribution) = bounds.(jd.marginals)