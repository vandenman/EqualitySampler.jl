using EqualitySampler, Turing, DynamicPPL, FillArrays, Plots

import	StatsBase 			as SB,
		LinearAlgebra 		as LA,
		StatsModels			as SM,
		DataFrames			as DF,
		GLM

import Logging
import ProgressMeter
import Serialization
import DataFrames: DataFrame
import StatsModels: @formula
import Suppressor
import Random

include("simulations/meansModel_Functions.jl")
include("simulations/helpersTuring.jl")
include("simulations/limitedLogger.jl")
include("simulations/customHMCAdaptation.jl")

@model function one_way_anova_mv_ss_submodel(obs_mean, obs_var, obs_n, Q, partition = nothing, ::Type{T} = Float64) where {T}

	n_groups = length(obs_mean)

	σ² 				~ JeffreysPriorVariance()
	μ_grand 		~ Normal(0, 1)

	# The setup for θ follows Rouder et al., 2012, p. 363
	g   ~ InverseGamma(0.5, 0.5)

	θ_r ~ MvNormal(n_groups - 1, 1.0)
	# ensure the sum to zero constraint
	θ_s = Q * (sqrt(g) .* θ_r)

	# constrain θ according to the sampled equalities
	θ_cs = isnothing(partition) ? θ_s : average_equality_constraints(θ_s, partition)

	# definition from Rouder et. al., (2012) eq 6.
	for i in 1:n_groups
		Turing.@addlogprob! _univariate_normal_likelihood(obs_mean[i], obs_var[i], obs_n[i], μ_grand + sqrt(σ²) * θ_cs[i], σ²)
	end
	return (θ_cs, )

end

@model function one_way_anova_mv_ss_eq_submodel(obs_mean, obs_var, obs_n, Q, partition_prior::D, ::Type{T} = Float64) where {T, D<:AbstractMvUrnDistribution}

	partition ~ partition_prior
	θ_cs = @submodel $(Symbol("one_way_anova_mv_ss_submodel")) one_way_anova_mv_ss_submodel(obs_mean, obs_var, obs_n, Q, partition, T)
	return (θ_cs, )

end

n_groups = 5
n_obs_per_group = 100
y, df, D, true_values = simulate_data_one_way_anova(n_groups, n_obs_per_group);
obs_mean, obs_var, obs_n = get_suff_stats(df)
Q = getQ_Rouder(n_groups)

iterations = 2_000

# full model

mod_full = one_way_anova_mv_ss_submodel(obs_mean, obs_var, obs_n, Q)
samps = sample(mod_full, NUTS(), 100)

mean_θ_cs_full, θ_cs_full, chain_full, model_full = fit_model(df, mcmc_iterations = iterations)
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
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq)) # inspect sampled equality constra


using Bijectors

@edit bijector(Gamma(1, 1))
