using EqualitySampler, Turing, DynamicPPL, FillArrays, Plots

import	StatsBase 			as SB,
		LinearAlgebra 		as LA,
		StatsModels			as SM,
		DataFrames			as DF,
		GLM

import Logging
import DataFrames: DataFrame
import StatsModels: @formula
import Random
import AdvancedMH

include("simulations/helpersTuring.jl")
include("simulations/silentGeneratedQuantities.jl")
include("simulations/meansModel_Functions.jl")
include("simulations/customHMCAdaptation.jl")
include("simulations/limitedLogger.jl")

n_groups = 5
n_obs_per_group = 10_000
# true_model = collect(1:n_groups)
true_model = reduce_model(sample_true_model(n_groups, 0))
true_θ = get_θ(0.2, true_model)

priors = (
	("uniform",		k->UniformMvUrnDistribution(k)),

	("betabinom11",	k->BetaBinomialMvUrnDistribution(k, 1, 1)),
	("betabinomk1",	k->BetaBinomialMvUrnDistribution(k, k, 1)),
	("betabinom1k",	k->BetaBinomialMvUrnDistribution(k, 1, k)),

	("dppalpha0.5",	k->RandomProcessMvUrnDistribution(k, Turing.RandomMeasures.DirichletProcess(0.5))),
	("dppalpha1",	k->RandomProcessMvUrnDistribution(k, Turing.RandomMeasures.DirichletProcess(1.0))),
	("dppalphak",	k->RandomProcessMvUrnDistribution(k, Turing.RandomMeasures.DirichletProcess(dpp_find_α(k))))
)

y, df, D, true_values = simulate_data_one_way_anova(n_groups, n_obs_per_group, true_θ, true_model, 0.0, 1.0);
get_suff_stats(df)

# (priorname, priorfun) = priors[2]
Logging.disable_logging(Logging.Debug)

# setadbackend(:reversediff)
function get_timings(priors)
	resultsDict = Dict()
	for (priorname, priorfun) in priors
		@show priorname
		Logging.with_logger(limited_warning_logger(3)) do
			resultsDict[priorname] = fit_model(df, mcmc_iterations = 10_000, mcmc_burnin = 5_000, partition_prior = priorfun(n_groups),
				hmc_stepsize = 0.00625,
				n_leapfrog = 10,
				custom_hmc_adaptation = false
				# custom_hmc = HMCDA(200, 0.65, 0.3, :μ_grand, :σ², :θ_r, :g)
			);
		end
	end
	return map(x->MCMCChains.wall_duration(x[3]), values(resultsDict)), resultsDict
end

times, resultsDict = get_timings(priors);
times

@btime logpdf(BetaBinomialMvUrnDistribution(n_groups, 1, n_groups), [1, 2, 3, 4, 5])
@btime logpdf(BetaBinomialMvUrnDistribution(n_groups, 1, 1), [1, 2, 3, 4, 5])
@btime logpdf(DirichletProcessMvUrnDistribution(n_groups, 0.5), [1, 2, 3, 4, 5])
dpp0 = DirichletProcessMvUrnDistribution(n_groups)
dpp1 = DirichletProcessMvUrnDistribution(n_groups, 1)
@btime logpdf(dpp1, [1, 2, 3, 4, 5])
@btime logpdf(dpp0, [1, 2, 3, 4, 5])

fit_model(df, mcmc_iterations = 3, mcmc_burnin = 1, partition_prior = UniformMvUrnDistribution(5),
				hmc_stepsize = 0.0,
				n_leapfrog = 10
				# custom_hmc = HMCDA(200, 0.65, 0.3, :μ_grand, :σ², :θ_r, :g)
			)

err_res = JLD2.load("debugging.jld2")

obs_mean, obs_n, obs_var, pop_mean, pop_var, result = values(sort(err_res))

result = -obs_n / 2.0 * (log(2pi) + log(pop_var)) - 1 / (2pop_var) * ((obs_n - 1) * obs_var + obs_n * (obs_mean - pop_mean)^2)
isinf(pop_mean)
isinf(pop_var)
isnan(pop_mean)
isnan(pop_var)
isnan(result)
isinf(result)

subplots = [plot(resultsDict[k][2]', legend = false, title = k) for k in keys(resultsDict)]
plot(subplots..., layout = (3, 2))

plot(resultsDict["betabinom11"][2]')

sort(compute_model_probs(resultsDict["betabinom11"][3]),  byvalue=true, rev=true)

resultsDict[priorname] = fit_model(df, mcmc_iterations = 10_000, mcmc_burnin = 5_000, partition_prior = priorfun(n_groups), use_Gibbs = true,
		hmc = HMCDA(20, 0.65, 0.3, :μ_grand, :σ², :θ_r, :g)
	);

# MCMCChains.wall_duration(resultsDict["betabinom11"][3])
map(x->MCMCChains.wall_duration(x[3]), values(resultsDict))

k = 5
m = generate_distinct_models(k)
prior_inst = map(x->x[2](k), priors)
for p in prior_inst
	@btime sum(logpdf(p, col) for col in eachcol(m))
end


#=

N = 10_000

bad:
0.0125

good:
0.003125
0.00625

=#


@model function one_way_anova_eq_mv_ss_2(obs_mean, obs_var, obs_n, Q, partition_prior::D, ::Type{T} = Float64) where {T, D<:AbstractMvUrnDistribution}

	n_groups = length(obs_mean)

	σ² 				~ InverseGamma(1, 1)
	μ_grand 		~ Normal(0, 1)

	# pass the prior like this to the model?
	partition ~ partition_prior

	# The setup for θ follows Rouder et al., 2012, p. 363
	g   ~ InverseGamma(0.5, 0.5)
	θ_r ~ filldist(Normal(0, sqrt(g)), n_groups - 1)
	# ensure the sum to zero constraint
	θ_s = Q * θ_r

	θ_cs = μ_grand .+ sqrt(σ²) .* θ_s

	for i in 1:n_groups
		obs_mean[i] ~ NormalSuffStat(obs_var[i], θ_cs[partition[i]], σ², obs_n[i])
	end
	return (θ_s[partition], )
end
obs_mean, obs_var, obs_n = get_suff_stats(df)
Q = getQ(length(obs_mean))

partition_prior = priors[2][2](n_groups)
model = one_way_anova_eq_mv_ss_2(obs_mean, obs_var, obs_n, Q, partition_prior)

function GammaMeanVar(m, v)
	# @show m, v
	if iszero(m)
		m += 0.001
	end
	θ = v / m
	Gamma(m / θ, θ)
end
spl = Gibbs(
	GibbsConditional(:partition, EqualitySampler.PartitionSampler(length(obs_mean), get_log_posterior(obs_mean, obs_var, obs_n, Q, partition_prior))),
	# HMC(0.0, 10, :μ_grand, :σ², :θ_r, :g)
	MH(
		:μ_grand			=> AdvancedMH.RandomWalkProposal(Normal(0.0, 0.25)),
		:g					=> x -> GammaMeanVar(x, 0.25),
		:σ²					=> x -> GammaMeanVar(x, 0.25),#AdvancedMH.RandomWalkProposal(Normal(0.0, 0.25)),
		# Symbol("θ_r[1]")	=> AdvancedMH.RandomWalkProposal(Normal(0, 0.25)),
		# Symbol("θ_r[2]")	=> AdvancedMH.RandomWalkProposal(Normal(0, 0.25)),
		# Symbol("θ_r[3]")	=> AdvancedMH.RandomWalkProposal(Normal(0, 0.25)),
		# Symbol("θ_r[4]")	=> AdvancedMH.RandomWalkProposal(Normal(0, 0.25)),
		:θ_r				=> AdvancedMH.RandomWalkProposal(MvNormal(length(obs_mean) - 1, 0.25))
	)
)

samples = sample(model, spl, 10_000)
# samples = sample(model, spl, 10_000; discard_initial = 2_000)
θ_cs = get_θ_cs(model, samples)
mean_θ_cs = vec(mean(θ_cs, dims = 2))
plot(θ_cs')
scatter(true_values[:θ], mean_θ_cs, legend = :none); Plots.abline!(1, 0)
scatter(true_values[:θ], mean_θ_cs, legend = :none); Plots.abline!(1, 0)

Qs = getQ_Stan.(2:k)
u = randn(k-1)

[u[1:i]*Qs[i] for i in 1:k-1]
[Qs[i]*u[1:i] for i in 1:k-1]

partition = [1, 2, 3, 4, 4]

u11 = Qs[4] * u
u12 = average_equality_constraints(Qs[4] * u, partition)
u13 = (Qs[3] * u[1:3])[partition]



sum(u11)
sum(u12)


@model function gdemo(x)
    s² ~ InverseGamma(2,3)
    m ~ MvNormal(length(x), sqrt(s²))
    x ~ MvNormal(m, sqrt(s²))
end

# Use a static proposal for s and random walk with proposal
# standard deviation of 0.25 for m.
x = collect(1.5:.5:9.5)
chain = sample(
    gdemo(x),
    MH(
        :s => AdvancedMH.StaticProposal(InverseGamma(2,3)),
        :m => AdvancedMH.RandomWalkProposal(MvNormal(length(x), 0.25))
		# Symbol("m[1]") => AdvancedMH.RandomWalkProposal(Normal(0, 0.25)),
		# Symbol("m[2]") => AdvancedMH.RandomWalkProposal(Normal(0, 0.25))
	),
	1_000
)
mean(chain)




@model function test_model_full(obs_mean, obs_var, obs_n, Q) where {T, D<:AbstractMvUrnDistribution}

	n_groups = length(obs_mean)

	σ² 				~ InverseGamma(1, 1)
	μ_grand 		~ Normal(0, 1)

	# The setup for θ follows Rouder et al., 2012, p. 363
	g   ~ InverseGamma(0.5, 0.5)

	# θ_r ~ MvNormal(n_groups - 1, 1.0)
	# ensure the sum to zero constraint
	# θ_s = Q * (sqrt(g) .* θ_r)

	# constrain θ according to the sampled equalities
	θ_cs = zeros(length(obs_mean))#θ_s#average_equality_constraints(θ_s, partition)

	# definition from Rouder et. al., (2012) eq 6.
	for i in 1:n_groups
		obs_mean[i] ~ NormalSuffStat(obs_var[i], μ_grand + sqrt(σ²) * θ_cs[i], σ², obs_n[i])
	end
	return (θ_cs, )

end

n_groups = 5
n_obs_per_group = 100
true_model = collect(1:n_groups)
# true_model = reduce_model(sample_true_model(n_groups, 0))
true_θ = get_θ(0.2, true_model)

_, df, D, true_values = simulate_data_one_way_anova(n_groups, n_obs_per_group, true_θ, true_model, 0.0, 1.0);
obs_mean, obs_var, obs_n = get_suff_stats(df)
Q = getQ(length(obs_mean))

model_full = test_model_full(obs_mean, obs_var, obs_n, Q)
spl_full = Gibbs(
	MH(
		:μ_grand			=> AdvancedMH.RandomWalkProposal(Normal(0.0, 0.25)),
		# :g					=> AdvancedMH.RandomWalkProposal(Gamma(.5, .5)),
		:σ²					=> AdvancedMH.RandomWalkProposal(Normal(0.0, .15)),
		# Symbol("θ_r[1]")	=> AdvancedMH.RandomWalkProposal(Normal(0, 0.25)),
		# Symbol("θ_r[2]")	=> AdvancedMH.RandomWalkProposal(Normal(0, 0.25)),
		# Symbol("θ_r[3]")	=> AdvancedMH.RandomWalkProposal(Normal(0, 0.25)),
		# Symbol("θ_r[4]")	=> AdvancedMH.RandomWalkProposal(Normal(0, 0.25)),
		# :θ_r				=> AdvancedMH.RandomWalkProposal(MvNormal(length(obs_mean) - 1, 0.25))
	)
)
samples_full = sample(model_full, spl_full, 1_000)



@model function test_model_eq(obs_mean, obs_var, obs_n, Q, partition_prior::D, ::Type{T} = Float64) where {T, D<:AbstractMvUrnDistribution}

	n_groups = length(obs_mean)

	σ² 				~ InverseGamma(1, 1)
	μ_grand 		~ Normal(0, 1)

	# pass the prior like this to the model?
	partition ~ partition_prior

	# The setup for θ follows Rouder et al., 2012, p. 363
	g   ~ InverseGamma(0.5, 0.5)

	θ_r ~ MvNormal(n_groups - 1, 1.0)
	# ensure the sum to zero constraint
	θ_s = Q * (sqrt(g) .* θ_r)

	# constrain θ according to the sampled equalities
	θ_cs = average_equality_constraints(θ_s, partition)

	# definition from Rouder et. al., (2012) eq 6.
	for i in 1:n_groups
		obs_mean[i] ~ NormalSuffStat(obs_var[i], μ_grand + sqrt(σ²) * θ_cs[i], σ², obs_n[i])
	end
	return (θ_cs, )

end

obs_mean, obs_var, obs_n = get_suff_stats(df)
Q = getQ(length(obs_mean))

model_eq = test_model_eq(obs_mean, obs_var, obs_n, Q, UniformMvUrnDistribution(length(obs_mean)))
spl_eq = Gibbs(
	GibbsConditional(:partition, EqualitySampler.PartitionSampler(length(obs_mean), get_log_posterior(obs_mean, obs_var, obs_n, Q, partition_prior))),
	# HMC(0.0, 10, :μ_grand, :σ², :θ_r, :g)
	MH(
		:μ_grand			=> AdvancedMH.RandomWalkProposal(Normal(0.0, 0.25)),
		:g					=> x -> GammaMeanVar(x, 0.25),
		:σ²					=> x -> GammaMeanVar(x, 0.25),
		Symbol("θ_r[1]")	=> AdvancedMH.RandomWalkProposal(Normal(0, 0.25)),
		Symbol("θ_r[2]")	=> AdvancedMH.RandomWalkProposal(Normal(0, 0.25)),
		Symbol("θ_r[3]")	=> AdvancedMH.RandomWalkProposal(Normal(0, 0.25)),
		Symbol("θ_r[4]")	=> AdvancedMH.RandomWalkProposal(Normal(0, 0.25))
		# :θ_r				=> AdvancedMH.RandomWalkProposal(MvNormal(length(obs_mean) - 1, 0.25))
	)
)
samples_eq = sample(model_eq, spl_eq, 50_000; discard_initial = 10_000)

θ_cs = get_θ_cs(model_eq, samples_eq)
mean_θ_cs = vec(mean(θ_cs, dims = 2))
plot(θ_cs')
scatter(true_values[:θ], mean_θ_cs, legend = :none); Plots.abline!(1, 0)
scatter(obs_mean, mean_θ_cs, legend = :none); Plots.abline!(1, 0)

spl_eq = Gibbs(
	GibbsConditional(:partition, EqualitySampler.PartitionSampler(length(obs_mean), get_log_posterior(obs_mean, obs_var, obs_n, Q, partition_prior))),
	HMC(0.0, 10, :μ_grand, :σ², :g),
	HMC(0.0, 10, :θ_r)
	# MH(
	# 	:μ_grand			=> AdvancedMH.RandomWalkProposal(Normal(0.0, 0.25)),
	# 	:g					=> x -> GammaMeanVar(x, 0.25),
	# 	:σ²					=> x -> GammaMeanVar(x, 0.25),
	# 	:θ_r				=> AdvancedMH.RandomWalkProposal(MvNormal(length(obs_mean) - 1, 0.25))
	# )
)
samples_eq = sample(model_eq, spl_eq, 10_000)

θ_cs = get_θ_cs(model_eq, samples_eq)
mean_θ_cs = vec(mean(θ_cs, dims = 2))
plot(θ_cs')
scatter(true_values[:θ], mean_θ_cs, legend = :none); Plots.abline!(1, 0)
scatter(obs_mean, mean_θ_cs, legend = :none); Plots.abline!(1, 0)

3 * 28_000 / 60 / 60

10 * 28_000 / 60 / 60 / 4

@edit HMC(0.0, 10, :μ_grand, :σ², :θ_r, :g)

import AdvancedHMC as AHMC
hamil = AHMC.Hamiltonian(AdvancedHMC.UnitEuclideanMetric, lπ, )

hmc_instance = AdvancedHMC.HMC{hamil, (:μ_grand, :σ², :θ_r, :g), AdvancedHMC.UnitEuclideanMetric}(0.0, 10)


@model function test_normal(x)
	sigma ~ Gamma(1, 1)
	mean ~ Normal()
	x ~ MvNormal(mean, sigma)
end



import ForwardDiff
f = x -> EqualitySampler._univariate_normal_likelihood(obs_mean[1], obs_var[1], obs_n[1], x[1], x[2]);
g = x -> ForwardDiff.gradient(f, x);
tt = vcat(true_values[:μ] .+ true_values[:σ] .* true_values[:θ])

u = vcat(tt[1], true_values[:σ])
f(u)
g(u)



ff(x) = sum(sin, x) + prod(tan, x) * sum(sqrt, x);
x = rand(5)
gg = x -> ForwardDiff.gradient(f, x)
ff(x)
gg(x)
