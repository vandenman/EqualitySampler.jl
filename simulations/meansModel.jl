# unfinished

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
include("simulations/meansModel_Functions.jl")

function show_code_warntype(model)
	@code_warntype model.f(
		Random.GLOBAL_RNG,
		model,
		Turing.VarInfo(model),
		Turing.SampleFromPrior(),
		Turing.DefaultContext(),
		model.args...,
	)
end

# example with full model, no equalities
n_groups = 6
n_obs_per_group = 100
y, df, D, true_values = simulate_data_one_way_anova(n_groups, n_obs_per_group);

# X = SM.modelmatrix(@formula(y ~ 1 + g).rhs, df, hints = Dict(:g => SM.FullDummyCoding()))
ests, mod = fit_lm(y, df)
scatter(true_values[:θ], ests, legend = :none); Plots.abline!(1, 0)

iterations = 2_000

# full model
mean_θ_cs_full, θ_cs_full, chain_full, model_full = fit_model(df, iterations = iterations)
plot(θ_cs_full') # trace plots
hcat(ests, mean_θ_cs_full) # compare frequentist estimates to posterior means
scatter(true_values[:θ], mean_θ_cs_full, legend = :none); Plots.abline!(1, 0)

# ideally we fix the type instability here
show_code_warntype(model_full)

# equality model with multivariate prior
mean_θ_cs_eq, θ_cs_eq, chain_eq, model_eq = fit_model(df, partition_prior = UniformMvUrnDistribution(n_groups))
plot(θ_cs_eq') # trace plots
hcat(ests, mean_θ_cs_eq) # compare frequentist estimates to posterior means
scatter(true_values[:θ], mean_θ_cs_eq, legend = :none); Plots.abline!(1, 0)
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq)) # inspect sampled equality constraints

show_code_warntype(model_eq)

# equality model with conditional distributions
mean_θ_cs_eq2, θ_cs_eq2, chain_eq2, model_eq2 = fit_model(df, partition_prior = BetaBinomialConditionalUrnDistribution(n_groups), iterations = iterations)
plot(θ_cs_eq2') # trace plots
hcat(ests, mean_θ_cs_eq2) # compare frequentist estimates to posterior means
scatter(true_values[:θ], mean_θ_cs_eq2, legend = :none); Plots.abline!(1, 0)
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq2))

# this one shows some additional type instabilities, maybe because it uses lambdas?
show_code_warntype(model_eq2)

partition_prior = BetaBinomialConditionalUrnDistribution(n_groups)
get_partition_prior(partition_prior, ones(Int, n_groups), 2)
@code_warntype get_partition_prior(partition_prior, 2, ones(Int, n_groups))

# example with equalities
n_groups = 6
n_obs_per_group = 100
θ_raw = randn(n_groups) .* 3
θ_raw .-= mean(θ_raw)
true_model = [1, 1, 2, 2, 3, 3]
θ_true = average_equality_constraints(θ_raw, true_model)
y, df, D, true_values = simulate_data_one_way_anova(n_groups, n_obs_per_group, θ_true);

# frequentist result
ests, mod = fit_lm(y, df)
scatter(true_values[:θ], ests, legend = :none);
Plots.abline!(1, 0)

# full model
mean_θ_cs_full, θ_cs_full, chain_full, model_full = fit_full_model(df)
plot(θ_cs_full') # trace plots
hcat(ests, mean_θ_cs_full) # compare frequentist estimates to posterior means
scatter(true_values[:θ], mean_θ_cs_full, legend = :none); Plots.abline!(1, 0)

# equality model with multivariate prior
mean_θ_cs_eq, θ_cs_eq, chain_eq, model_eq = fit_model(df, model_type = :eq_mv, prior_type = :uniform)
plot(θ_cs_eq') # trace plots
hcat(ests, mean_θ_cs_eq) # compare frequentist estimates to posterior means
scatter(true_values[:θ], mean_θ_cs_eq, legend = :none); Plots.abline!(1, 0)
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq)) # inspect sampled equality constraints

mean_θ_cs_eq, θ_cs_eq, chain_eq, model_eq = fit_model(df, model_type = :eq_mv, prior_type = :beta_binomial, prior_args = (bb_α = 2.0, bb_β = 2.0, dpp_α = 1.887))
plot(θ_cs_eq') # trace plots
hcat(ests, mean_θ_cs_eq) # compare frequentist estimates to posterior means
scatter(true_values[:θ], mean_θ_cs_eq, legend = :none); Plots.abline!(1, 0)
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq)) # inspect sampled equality constraints

mean_θ_cs_eq, θ_cs_eq, chain_eq, model_eq = fit_model(df, model_type = :eq_mv, prior_type = :dirichlet_process)
plot(θ_cs_eq') # trace plots
hcat(ests, mean_θ_cs_eq) # compare frequentist estimates to posterior means
scatter(true_values[:θ], mean_θ_cs_eq, legend = :none); Plots.abline!(1, 0)
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq)) # inspect sampled equality constraints

# equality model with conditional distributions
mean_θ_cs_eq, θ_cs_eq, chain_eq, model_eq = fit_model(df, model_type = :eq_cond, prior_type = :beta_binomial, prior_args = (bb_α = 2.0, bb_β = 2.0, dpp_α = 1.887))
plot(θ_cs_eq') # trace plots
hcat(ests, mean_θ_cs_eq) # compare frequentist estimates to posterior means
scatter(true_values[:θ], mean_θ_cs_eq, legend = :none); Plots.abline!(1, 0)
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq)) # inspect sampled equality constraints


mean_θ_cs_eq, θ_cs_eq, chain_eq, model_eq = fit_model2(df, model_type = :eq_mv, partition_prior = UniformMvUrnDistribution(n_groups))
plot(θ_cs_eq') # trace plots
hcat(ests, mean_θ_cs_eq) # compare frequentist estimates to posterior means
scatter(true_values[:θ], mean_θ_cs_eq, legend = :none); Plots.abline!(1, 0)
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq)) # inspect sampled equality constraints

# example with full model, no equalities
n_groups = 20
n_obs_per_group = 100
true_model = sample_true_model(20, 80)
true_θ = get_θ(0.2, true_model)

y, df, D, true_values = simulate_data_one_way_anova(n_groups, n_obs_per_group, true_θ);

resMv = fit_model(df, mcmc_iterations = 50_000, mcmc_burnin = 5_000, partition_prior = BetaBinomialMvUrnDistribution(1, 1.0, 1.0))
mean_θ_cs_eq, θ_cs_eq, chain_eq, model_eq = resMv
plot(θ_cs_eq') # trace plots
hcat(ests, mean_θ_cs_eq) # compare frequentist estimates to posterior means
scatter(true_values[:θ], mean_θ_cs_eq, legend = :none); Plots.abline!(1, 0)
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq)) # inspect sampled equality constraints
rtrue_model = reduce_model(true_model)
LA.UnitLowerTriangular([i == j for i in rtrue_model, j in rtrue_model])
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq) .> 0.5) # inspect sampled equality constraints

resConditional = fit_model(df, mcmc_iterations = 10_000, mcmc_burnin = 5_000, partition_prior = BetaBinomialConditionalUrnDistribution(n_groups, 1.0, 1.0))
mean_θ_cs_eq, θ_cs_eq, chain_eq, model_eq = resConditional

plot(θ_cs_eq') # trace plots
hcat(ests, mean_θ_cs_eq) # compare frequentist estimates to posterior means
scatter(true_values[:θ], mean_θ_cs_eq, legend = :none); Plots.abline!(1, 0)
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq)) # inspect sampled equality constraints
rtrue_model = reduce_model(true_model)
LA.UnitLowerTriangular([i == j for i in rtrue_model, j in rtrue_model])
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq) .> 0.5) # inspect sampled equality constraints

# GibbsConditional attempt
n_groups = 6
n_obs_per_group = 1000
true_model = reduce_model(sample_true_model(n_groups, 50))
true_θ = get_θ(0.2, true_model)

y, df, D, true_values = simulate_data_one_way_anova(n_groups, n_obs_per_group, true_θ);


partition_prior = UniformConditionalUrnDistribution(n_groups)
obs_mean, obs_var, obs_n = get_suff_stats(df)
Q = getQ(length(obs_mean))
model, sampler0 = get_model_and_sampler(partition_prior, obs_mean, obs_var, obs_n, Q)

function get_log_posterior(obs_mean, obs_var, obs_n, Q)
	return function logposterior(nextValues, c)
		σ = sqrt(c.σ²)
		θ_s = Q * c.θ_r
		θ_cs = average_equality_constraints(θ_s, nextValues)
		return sum(logpdf(NormalSuffStat(obs_var[j], c.μ_grand + θ_cs[j], σ, obs_n[j]), obs_mean[j]) for j in 1:n_groups)
	end
end
tt = get_log_posterior(obs_mean, obs_var, obs_n, Q)


sampler1 = Gibbs(GibbsConditional(:equal_indices, EqualitySampler.PartitionSampler(n_groups, get_log_posterior(obs_mean, obs_var, obs_n, Q))), HMC(0.01, 10, :μ_grand, :σ², :θ_r))
sample(model, sampler1, 5000)

resOriginal   = fit_model(df, mcmc_iterations = 25_000, mcmc_burnin = 5_000, partition_prior = UniformMvUrnDistribution(n_groups), use_Gibbs = false)
resGibbsStuff = fit_model(df, mcmc_iterations = 25_000, mcmc_burnin = 5_000, partition_prior = UniformMvUrnDistribution(n_groups), use_Gibbs = true)

mean_θ_cs_eq, θ_cs_eq, chain_eq, model_eq = resGibbsStuff
plot(θ_cs_eq') # trace plots
hcat(ests, mean_θ_cs_eq) # compare frequentist estimates to posterior means
scatter(true_values[:θ], mean_θ_cs_eq, legend = :none); Plots.abline!(1, 0)
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq)) # inspect sampled equality constraints
rtrue_model = reduce_model(true_model)
LA.UnitLowerTriangular([i == j for i in rtrue_model, j in rtrue_model])
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq) .> 0.5) # inspect sampled equality constraints



# quick testing
n_groups = 3
n_obs_per_group = 100
y, df, D, true_values = simulate_data_one_way_anova(n_groups, n_obs_per_group);

iterations = 500
res1 = fit_model(df, iterations = iterations)
res2 = fit_model(df, iterations = iterations, partition_prior = UniformMvUrnDistribution(1))
res3 = fit_model(df, iterations = iterations, partition_prior = BetaBinomialMvUrnDistribution(1, 3, 2))
res4 = fit_model(df, iterations = iterations, partition_prior = RandomProcessMvUrnDistribution(1, Turing.RandomMeasures.DirichletProcess(3)))

tmp = fit_model(df, mcmc_iterations = 10, mcmc_burnin = 5, partition_prior = BetaBinomialConditionalUrnDistribution(n_groups, 1.0, 1.0))
show_code_warntype(tmp[4])
@code_warntype rand(BetaBinomialConditionalUrnDistribution(n_groups, 1.0, 1.0))
@code_warntype logpdf(BetaBinomialConditionalUrnDistribution(n_groups, 1.0, 1.0), 1)

@code_warntype EqualitySampler._pdf(BetaBinomialConditionalUrnDistribution(n_groups, 1.0, 1.0))

D = BetaBinomialConditionalUrnDistribution(n_groups, 1.0, 1.0)
complete_urns = D.urns
index = D.index
result = zeros(Float64, length(complete_urns))
@code_warntype EqualitySampler._pdf_helper!(result, D, complete_urns, index)


@code_warntype fit_model(df, iterations = iterations)



# TODO: there shouldn't be any red stuff, but there is :/ (this is mainly optimization though)
@model function model_test_1(obs_mean, obs_var, obs_n, Q, ::Type{T} = Float64) where {T}

	n_groups = length(obs_mean)

	σ² 				~ InverseGamma(1, 1)
	μ_grand 		~ Normal(0, 1)
	equal_indices 	~ UniformMvUrnDistribution(n_groups)

	# The setup for θ follows Rouder et al., 2012, p. 363
	θ_r ~ filldist(Normal(0, 1), n_groups - 1)
	# ensure that subtract the mean of θ to ensure the sum to zero constraint
	θ_s = Q * θ_r
	# # constrain θ according to the sampled equalities
	θ_cs = average_equality_constraints(θ_s, equal_indices)

	# # definition from Rouder et. al., (2012) eq 6.
	σ = sqrt(σ²)
	for i in 1:n_groups
		obs_mean[i] ~ NormalSuffStat(obs_var[i], μ_grand + θ_cs[i], σ, obs_n[i])
	end
	# return (θ_cs, )
end

inst_test_1 = model_test_1(obs_mean, obs_var, obs_n, Q)
show_code_warntype(inst_test_1)


import Random
d = UniformMvUrnDistribution(20)
u = rand(d)
@btime Distributions._rand!($Random.GLOBAL_RNG, $d, $u)
@edit Distributions._rand!(Random.GLOBAL_RNG, d, u)

VSCodeServer.@profview [Distributions._rand!(Random.GLOBAL_RNG, d, u) for _ in 1:100]
ProfileView.@profview [Distributions._rand!(Random.GLOBAL_RNG, d, u) for _ in 1:100]

n_groups = 6
n_obs_per_group = 100
θ_raw = randn(n_groups) .* 3
θ_raw .-= mean(θ_raw)
true_model = [1, 1, 2, 2, 3, 3]
θ_true = average_equality_constraints(θ_raw, true_model)
y, df, D, true_values = simulate_data_one_way_anova(n_groups, n_obs_per_group, θ_true);

obs_mean, obs_var, obs_n = get_suff_stats(df)

@code_warntype EqualitySampler.NormalSuffStat(obs_var[1], obs_mean[1], obs_var[1], obs_n[1])
dd = EqualitySampler.NormalSuffStat(obs_var[1], obs_mean[1], obs_var[1], obs_n[1])
@code_warntype logpdf(dd, obs_mean[1])
@btime logpdf(dd, obs_mean[1])


function test_pdf_1(θ_i, θ_other, mean = 0.0)
	sd = sqrt(1 / sum(x-> (θ_i - x)^2, θ_other))
	return pdf(Normal(mean, sd), θ_i)
end

distance(x, y) = 1 / (x - y)^2
function test_pdf_2(θ_i, θ_other, mean = 0.0, sd = 1.0, sd2 = 0.25)
	weights = distance.(θ_i, [0; θ_other])
	weights ./= sum(weights)

	val = weights[1] * pdf(Normal(mean, sd), θ_i)
	for i in eachindex(θ_other)
		val += weights[i+1] * pdf(Normal(θ_other[i], sd2), θ_i)
	end
	return val
end



θ_other = θ_cs_full[:, 4]#[1, -2]
xvals = range(-3, stop = 3.0, length = 2^10)
yvals_1 = test_pdf_1.(xvals, Ref(θ_other))
yvals_2 = test_pdf_2.(xvals, Ref(θ_other))

plot(xvals, yvals_1)
plot(xvals, hcat(yvals_1, yvals_2))

