using UncertaintyQuantification

mc = MonteCarlo(10^4)
LHC = LatinHypercubeSampling(10^4)
ss = SubSetSimulation(2000, 0.1, 10, Uniform(-0.2, 0.2))
ss_infinity = SubSetInfinity(2000, 0.1, 10, 0.5)
ss_infinity_adaptive = SubSetInfinityAdaptive(2000, 0.1, 10, 10)
LS = LineSampling(200)

## Alvarez model, from 10^6 interval MC he gets pf = [2.590 × 10−4, 0.503] 

X1 = ProbabilityBox{Normal}([Interval(-1, 2, :μ), Parameter(1, :σ)], :X1) 
X2 = ProbabilityBox{Normal}([Interval(-2, 1, :μ), Parameter(2, :σ)], :X2) 

inputs = [X1, X2]
models = Model(df -> df.X1.^2 .+ df.X2, :g)
performance(df) = 7 .- df.g

@time interval_form = probability_of_failure(models, performance, inputs, IntervalMonteCarlo(FORM()))
@time interval_mc = probability_of_failure(models, performance, inputs, IntervalMonteCarlo(mc))
@time interval_lhc = probability_of_failure(models, performance, inputs, IntervalMonteCarlo(LHC))
@time interval_ss = probability_of_failure(models, performance, inputs, IntervalMonteCarlo(ss))
@time interval_ss_infinity = probability_of_failure(models, performance, inputs, IntervalMonteCarlo(ss_infinity))
@time interval_ss_infinity_adaptive = probability_of_failure(models, performance, inputs, IntervalMonteCarlo(ss_infinity_adaptive))

@time interval_line_sampling = probability_of_failure(models, performance, inputs, IntervalMonteCarlo(LS))
