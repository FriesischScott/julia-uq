struct InputStandardizationGP{R<:Real}
    min::R
    max::R
end

function to_standard(x::Float64, min::Float64, max::Float64)
    return (x - min) / (max - min)
end

function to_physical(x::Float64, min::Float64, max::Float64)
    return (max - min) * x + min
end
