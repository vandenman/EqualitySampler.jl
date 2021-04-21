# unfinished

using EqualitySampler, Turing, DynamicPPL, FillArrays, Plots

import	StatsBase 			as SB,
		LinearAlgebra 		as LA,
		StatsModels			as SM,
		DataFrames			as DF,
		GLM

import DataFrames: DataFrame, categorical
import StatsModels: @formula
import Suppressor
import Random

include("simulations/plotFunctions.jl")
include("simulations/helpersTuring.jl")

function simulate_data_one_way_anova(
		n_groups::Integer,
		n_obs_per_group::Integer,
		θ::Vector{Float64} = Float64[],
		μ::Real = 0.0,
		σ::Real = 1.0)

	if isempty(θ)
		θ = 2 .* randn(n_groups)
	end
	@assert length(θ) == n_groups

	n_obs = n_groups * n_obs_per_group
	θ2 = θ .- SB.mean(θ)

	g = Vector{Int}(undef, n_obs)
	for (i, r) in enumerate(Iterators.partition(1:n_obs, ceil(Int, n_obs / n_groups)))
		g[r] .= i
	end

	D = MvNormal(μ .+ σ .* θ[g], σ)
	y = rand(D)

	df = DataFrame(:y => y, :g => categorical(g))
	# combine(groupby(df, :g), :y => mean, :y => x->mean(x .- μ_grand_true))

	true_values = Dict(
		:σ			=> σ,
		:μ			=> μ,
		:θ			=> θ2,
	)

	return y, df, D, true_values

end

function fit_lm(y, df)
	mod = SM.fit(GLM.LinearModel, @formula(y ~ 1 + g), df)#, contrasts = Dict(:g => StatsModels.FullDummyCoding()))
	# transform the coefficients to a grand mean and offsets
	coefs = copy(GLM.coef(mod))
	ests = similar(coefs)
	ests[1] = coefs[1] - mean(df[!, :y])
	ests[2:end] .= coefs[2:end] .+ coefs[1] .- mean(df[!, :y])
	return ests, mod
end

function getQ(n_groups)::Matrix{Float64}
	# X = StatsModels.modelmatrix(@formula(y ~ 0 + g).rhs, DataFrame(:g => g), hints = Dict(:g => StatsModels.FullDummyCoding()))
	Σₐ = Matrix{Float64}(LA.I, n_groups, n_groups) .- (1.0 / n_groups)
	_, v::Matrix{Float64} = LA.eigen(Σₐ)
	v[end:-1:1, end:-1:2] # this is what happens in Rouder et al., (2012) eq ...
end

@model function one_way_anova_full_ss(obs_mean, obs_var, obs_n, Q, ::Type{T} = Float64) where {T}

	n_groups = length(obs_mean)
	σ² ~ InverseGamma(1, 1)
	μ_grand ~ Normal(0, 1)

	# The setup for θ follows Rouder et al., 2012, p. 363
	θ_r ~ filldist(Normal(0, 1), n_groups - 1)
	# ensure that subtract the mean of θ to ensure the sum to zero constraint
	θ_s = Q * θ_r
	θ_cs = θ_s # full model

	# definition from Rouder et. al., (2012) eq 6.
	σ = sqrt(σ²)
	for i in 1:n_groups
		obs_mean[i] ~ NormalSuffStat(obs_var[i], μ_grand + θ_cs[i], σ, obs_n[i])
	end
	return (θ_cs, )
end

function get_suff_stats(df)

	temp_df::DataFrame = DF.combine(DF.groupby(df, :g), :y => mean, :y => x -> mean(xx -> xx^2, x), :y => length)

	obs_mean::Vector{Float64}	= temp_df[!, "y_mean"]
	obs_var::Vector{Float64}	= temp_df[!, "y_function"]
	obs_n::Vector{Int}			= temp_df[!, "y_length"]

	return obs_mean, obs_var, obs_n
end

function get_θ_cs(model, chain)
	gen = Suppressor.@suppress generated_quantities(model, chain)
	θ_cs = Matrix{Float64}(undef, length(first(gen[1])), length(gen))
	for i in eachindex(gen)
		θ_cs[:, i] = gen[i][1]
	end
	return θ_cs
end

function fit_full_model(df; iterations::Int = 15_000)

	obs_mean, obs_var, obs_n = get_suff_stats(df)

	Q = getQ(length(obs_mean))
	@assert isapprox(sum(Q * randn(length(obs_mean) - 1)), 0.0, atol = 1e-8)

	model = one_way_anova_full_ss(obs_mean, obs_var, obs_n, Q)
	# chain = sample(model, HMC(0.01, 10), 15_000)
	chain = sample(model, NUTS(), iterations)
	θ_cs = get_θ_cs(model, chain)
	mean_θ_cs = mean(θ_cs, dims = 2)
	return mean_θ_cs, θ_cs, chain, model

end


# this is very type unstable...
# function fit_model(df; which_model::Symbol = :full, iterations::Int = 15_000)

# 	obs_mean, obs_var, obs_n = get_suff_stats(df)

# 	Q = getQ(length(obs_mean))
# 	@assert isapprox(sum(Q * randn(length(obs_mean) - 1)), 0.0, atol = 1e-8)

# 	if which_model == :full
# 		model = one_way_anova_full_ss(obs_mean, obs_var, obs_n, Q)
# 		sampler = Nuts()
# 	elseif which_model ==:EqualitySampler
# 		model = one_way_anova_eq_ss(obs_mean, obs_var, obs_n, Q)
# 		spl = Gibbs(PG(20, :equal_indices), HMC(0.01, 10, :μ_grand, :σ², :θ_r))
# 	end
# 	chain = sample(model, sampler, iterations)
# 	θ_cs = get_θ_cs(model, chain)
# 	mean_θ_cs = mean(θ_cs, dims = 2)

# 	return mean_θ_cs, θ_cs, chain, model
# end

function average_equality_constraints(ρ::AbstractVector{<:Real}, equal_indices::AbstractVector{<:Integer})
	ρ_c = similar(ρ)
	# this can be done more efficiently but I'm not sure it matters when length(ρ) is small
	for i in eachindex(ρ)
		ρ_c[i] = mean(ρ[equal_indices .== equal_indices[i]])
	end
	return ρ_c
end

@model function one_way_anova_eq_ss(obs_mean, obs_var, obs_n, Q, ::Type{T} = Float64) where {T}

	n_groups = length(obs_mean)

	σ² 				~ InverseGamma(1, 1)
	μ_grand 		~ Normal(0, 1)
	equal_indices 	~ UniformMvUrnDistribution(n_groups)

	# The setup for θ follows Rouder et al., 2012, p. 363
	θ_r ~ filldist(Normal(0, 1), n_groups - 1)
	# ensure that subtract the mean of θ to ensure the sum to zero constraint
	θ_s = Q * θ_r
	# constrain θ according to the sampled equalities
	θ_cs = average_equality_constraints(θ_s, equal_indices)

	# definition from Rouder et. al., (2012) eq 6.
	σ = sqrt(σ²)
	for i in 1:n_groups
		obs_mean[i] ~ NormalSuffStat(obs_var[i], μ_grand + θ_cs[i], σ, obs_n[i])
	end
	return (θ_cs, )
end

function fit_eq_model(df; iterations::Int = 15_000)

	obs_mean, obs_var, obs_n = get_suff_stats(df)

	Q = getQ(length(obs_mean))
	@assert isapprox(sum(Q * randn(length(obs_mean) - 1)), 0.0, atol = 1e-8)

	model = one_way_anova_eq_ss(obs_mean, obs_var, obs_n, Q)
	spl = Gibbs(PG(20, :equal_indices), HMC(0.01, 10, :μ_grand, :σ², :θ_r))
	chain = sample(model, spl, iterations)
	θ_cs = get_θ_cs(model, chain)
	mean_θ_cs = mean(θ_cs, dims = 2)

	return mean_θ_cs, θ_cs, chain, model
end

# X = StatsModels.modelmatrix(@formula(y ~ 0 + g).rhs, DataFrame(:g => g), hints = Dict(:g => StatsModels.FullDummyCoding()))

n_groups = 6
n_obs_per_group = 100
y, df, D, true_values = simulate_data_one_way_anova(n_groups, n_obs_per_group);

# X = SM.modelmatrix(@formula(y ~ 1 + g).rhs, df, hints = Dict(:g => SM.FullDummyCoding()))
ests, mod = fit_lm(y, df)
scatter(true_values[:θ], ests, legend = :none);
Plots.abline!(1, 0)


mean_θ_cs_full, θ_cs_full, chain_full, model_full = fit_full_model(df)

plot(θ_cs_full')

scatter(true_values[:θ], mean_θ_cs_full, legend = :none);
Plots.abline!(1, 0)


mean_θ_cs_eq, θ_cs_eq, chain_eq, model_eq = fit_eq_model(df)
plot(θ_cs_eq')

scatter(true_values[:θ], mean_θ_cs_eq, legend = :none)
Plots.abline!(1, 0)

# inspect sampled equality constraints
post_model_probs = sort(compute_model_probs(chain_eq), byvalue=true)
post_model_probs_nonzero = filter(x->!iszero(x[2]), post_model_probs)
sort(post_model_probs_nonzero, rev=true)
compute_incl_probs(chain_eq)

model_counts = compute_model_counts(chain_eq)
filter(x->!iszero(x[2]), model_counts)
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq))


# example with equalities
n_groups = 6
n_obs_per_group = 200
θ_raw = randn(n_groups) .* 3
θ_raw .-= mean(θ_raw)
true_model = [1, 1, 2, 2, 3, 3]
θ_true = average_equality_constraints(θ_raw, true_model)
y, df, D, true_values = simulate_data_one_way_anova(n_groups, n_obs_per_group, θ_true);

mean_θ_cs_full, θ_cs_full, chain_full, model_full = fit_full_model(df)

plot(θ_cs_full')
scatter(true_values[:θ], mean_θ_cs_full, legend = :none);
Plots.abline!(1, 0)

mean_θ_cs_eq, θ_cs_eq, chain_eq, model_eq = fit_eq_model(df, iterations = 30_000)
plot(θ_cs_eq')

scatter(true_values[:θ], mean_θ_cs_eq, legend = :none)
Plots.abline!(1, 0)

# inspect sampled equality constraints
post_model_probs = sort(compute_model_probs(chain_eq), byvalue=true)
post_model_probs_nonzero = filter(x->!iszero(x[2]), post_model_probs)
sort(post_model_probs_nonzero, rev=true)
compute_incl_probs(chain_eq)

model_counts = compute_model_counts(chain_eq)
filter(x->!iszero(x[2]), model_counts)
LA.UnitLowerTriangular(compute_post_prob_eq(chain_eq))







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

NormalSuffStat <: Distribution