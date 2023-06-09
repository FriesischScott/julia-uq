struct TMCMC <: AbstractBayesianMethod # Transitional Markov Chain Monte Carlo
    snum::Int64
    γ::Real
    TMCMC(snum) = snum > 0 ? new(snum) : error("n must be greater than zero")
end

struct GibbsSampler <: AbstractBayesianMethod
    snum::Int64
    GibbsSampler(snum) = snum > 0 ? new(snum) : error("n must be greater than zero")
end

struct AdaptiveMetropolisHastings <: AbstractBayesianMethod # Adaptive Metropolis-Hastings
    snum::Integer # number of samples per chain
    cnum::Integer # number of chains
    x0::DataFrame # starting point
    C0::AbstractMatrix{<:Real} # initial proposal covariance


    function AdaptiveMetropolisHastings(snum::Integer,cnum::Integer,x0::DataFrame,C0::AbstractMatrix{<:Real})
        errChck = (size(x0,2) != size(C0,1))
        errChck <<= 1
        errChck += (size(C0,1) != size(C0,2))
        if snum<=0
            error("Number of samples <= 0!")
        end

        if errChck>0
            msg = ""
            if (errChck & 1) == 1
                msg = string(msg, length(msg)>0 ? " " : "" ,"Σ is not square!")
            end
            errChck >>= 1
            if (errChck & 1) == 1
                msg = string(msg, length(msg)>0 ? " " : "" ,"Dimension mismatch in μ and Σ!")
            end
            error(msg)
        end
        
        return new(snum,cnum,x0,C0)
    end
end

function bayesianupdating(
    loglikelihood::Function,
    prior::Function,
    amh::AdaptiveMetropolisHastings
)
    tar(x) = exp.(loglikelihood(x)).*prior(x)

    x0 = [copy(amh.x0) for _ in 1:amh.cnum]
    pmap(i -> AdaptiveMetropolisHastingsSingle!(tar,amh.snum,x0[i],amh.C0),1:amh.cnum)

    out = DataFrame()

    for i in 1:amh.cnum
        append!(out,x0[i])
    end

    return out
end

# TMCMC implementation
function bayesianupdating(
    loglikelihood::Function,
    prior::Function,
    tmcmc::TMCMC
)
    error("Not implemented!")

end

# GibbsSampler implementation
function bayesianupdating(
    loglikelihood::Function,
    prior::Function,
    gibbs::GibbsSampler
)

    error("Not implemented!")

end

# Adaptive Metropolis-Hastings, algorithm implemented according to Haario
# et al., "An Adaptive Metropolis Algorithm", https://doi.org/10.2307/3318737
function AdaptiveMetropolisHastingsSingle!(
    tar::Function,
    N::Integer,
    x0::DataFrame,
    C0::Matrix{<:Real},
    t0::Int64 = 1,
    s0::Float64 = 0.0001,
    burnin::Integer = 0,
)
    acc = 0 # accepted Samples

    for nS in 2:N+burnin
        if nS < t0 || acc==0
            C_ = C0
        else
            C_ = cov(Matrix(x0)) + s0 * C0
        end
        propD = Distributions.MvNormal(Vector(x0[nS-1,:]),C_)
        propX = DataFrame(names(x0) .=> rand(propD))
        lastPost = tar(x0[nS-1,:])
        propPost = tar(propX)
        #lastPost and propPost are always one-entry vectors, there should be no issue with always using the first entry?
        alpha = propPost[1] / lastPost[1] 
        if alpha > 1. || alpha > rand(1)[1]
            append!(x0,propX)
            acc += 1
        else
            push!(x0,x0[end,:])
        end
    end
    deleteat!(x0,1:burnin)

    return nothing

end

