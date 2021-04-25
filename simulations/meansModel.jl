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
include("simulations/anovaFunctions.jl")

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


# quick testing
n_groups = 3
n_obs_per_group = 100
y, df, D, true_values = simulate_data_one_way_anova(n_groups, n_obs_per_group);

iterations = 500
res1 = fit_model(df, iterations = iterations)
res2 = fit_model(df, iterations = iterations, partition_prior = UniformMvUrnDistribution(1))
res3 = fit_model(df, iterations = iterations, partition_prior = BetaBinomialMvUrnDistribution(1, 3, 2))
res4 = fit_model(df, iterations = iterations, partition_prior = RandomProcessMvUrnDistribution(1, Turing.RandomMeasures.DirichletProcess(3)))

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