function isimprecise(inputs::AbstractVector{<:UQInput})
    return any(isimprecise.(inputs))
end

function isimprecise(input::UQInput)
    return isa(input, ImpreciseUQInput)
end

function isimprecise(jd::JointDistribution)
    return any(isa.(jd.marginals, ImpreciseUQInput))
end