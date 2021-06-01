using EqualitySampler, Turing, DynamicPPL, FillArrays, Plots

import	StatsBase 			as SB,
		LinearAlgebra 		as LA,
		StatsModels			as SM,
		DataFrames			as DF,
		GLM

import DataFrames: DataFrame
import StatsModels: @formula
import Suppressor
import Random

# include("simulations/plotFunctions.jl") # <- unused?
include("simulations/helpersTuring.jl")
include("simulations/silentGeneratedQuantities.jl")
include("simulations/meansModel_Functions.jl")

# 1_000
# 0.01

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

	("dppalpha1",	k->RandomProcessMvUrnDistribution(k, Turing.RandomMeasures.DirichletProcess(1.0))),
	("dppalphak",	k->RandomProcessMvUrnDistribution(k, Turing.RandomMeasures.DirichletProcess(dpp_find_α(k))))
)

y, df, D, true_values = simulate_data_one_way_anova(n_groups, n_obs_per_group, true_θ, true_model, 0.0, 1.0);
get_suff_stats(df)

(priorname, priorfun) = priors[2]

resultsDict = Dict()
for (priorname, priorfun) in priors
	@show priorname
	resultsDict[priorname] = fit_model(df, mcmc_iterations = 10_000, mcmc_burnin = 5_000, partition_prior = priorfun(n_groups), use_Gibbs = true,
		hmc_stepsize = 0.0,
		n_leapfrog = 10
	);
end

subplots = [plot(resultsDict[k][2]', legend = false, title = k) for k in keys(resultsDict)]
plot(subplots..., layout = (3, 2))

plot(resultsDict["betabinom11"][2]')

sort(compute_model_probs(resultsDict["betabinom11"][3]),  byvalue=true, rev=true)

resultsDict[priorname] = fit_model(df, mcmc_iterations = 10_000, mcmc_burnin = 5_000, partition_prior = priorfun(n_groups), use_Gibbs = true,
		hmc_stepsize = 0.0,
		n_leapfrog = 10
	);

@edit Turing.setadbackend(:test)

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
spl = sampler = Gibbs(
	GibbsConditional(:partition, EqualitySampler.PartitionSampler(length(obs_mean), get_log_posterior(obs_mean, obs_var, obs_n, Q, partition_prior))),
	HMC(0.0, 10, :μ_grand, :σ², :θ_r, :g)
	# MH(
	# 	:μ_grand	=> AdvancedMH.RandomWalkProposal(Normal(0.0, 0.25)),
	# 	:θ_r		=> AdvancedMH.RandomWalkProposal(MvNormal(length(obs_mean) - 1, 0.25)),
	# 	:σ²			=> AdvancedMH.RandomWalkProposal(Truncated(Normal(0, 0.5), 0.0, Inf)),
	# 	:g			=> AdvancedMH.RandomWalkProposal(Truncated(Normal(0, 0.5), 0.0, Inf))
	# )
)

samples = sample(model, spl, 10_000; discard_initial = 5_000)
