import	StatsBase 			as SB,
		LinearAlgebra 		as LA,
		StatsModels			as SM,
		DataFrames			as DF,
		GLM

import CategoricalArrays
using Turing

function simulate_data_one_way_anova(
		n_groups::Integer,
		n_obs_per_group::Integer,
		θ::Vector{Float64} = Float64[],
		true_partition = collect(1:n_groups),
		μ::Real = 0.0,
		σ::Real = 1.0)

	if isempty(θ)
		# θ = 2 .* randn(n_groups)
		θ = 0.2 .* true_partition
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

	df = DF.DataFrame(:y => y, :g => CategoricalArrays.categorical(g))

	true_values = Dict(
		:σ			=> σ,
		:μ			=> μ,
		:θ			=> θ2,
	)

	return y, df, D, true_values

end

function fit_lm(y, df)
	mod = SM.fit(GLM.LinearModel, SM.@formula(y ~ 1 + g), df)#, contrasts = Dict(:g => StatsModels.FullDummyCoding()))
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
	Q = v[end:-1:1, end:-1:2] # this is what happens in Rouder et al., (2012) eq ...

	@assert isapprox(sum(Q * randn(n_groups-1)), 0.0, atol = 1e-8)

	return Q
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

	temp_df::DF.DataFrame = DF.combine(DF.groupby(df, :g), :y => mean, :y => x -> mean(xx -> xx^2, x), :y => length)

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

# function average_equality_constraints(ρ::AbstractVector{<:Real}, equal_indices::AbstractVector{<:Integer})
# 	ρ_c = similar(ρ)
# 	# this can be done more efficiently but I'm not sure it matters when length(ρ) is small
# 	for i in eachindex(ρ)
# 		ρ_c[i] = mean(ρ[equal_indices .== equal_indices[i]])
# 	end
# 	# for i in unique(equal_indices)
# 	# 	ρ_c[i] = mean(ρ[equal_indices .== equal_indices[i]])
# 	# end
# 	return ρ_c
# end

function average_equality_constraints(ρ::AbstractVector{<:Real}, equal_indices::AbstractVector{<:Integer})

	idx_vecs = [Int[] for _ in eachindex(equal_indices)]
	@inbounds for i in eachindex(equal_indices)
		push!(idx_vecs[equal_indices[i]], i)
	end

	ρ_c = similar(ρ)
	@inbounds for idx in idx_vecs
		isempty(idx) && continue
		ρ_c[idx] .= mean(ρ[idx])
	end

	return ρ_c
end

@model function one_way_anova_eq_mv_ss(obs_mean, obs_var, obs_n, Q, partition_prior::D, ::Type{T} = Float64) where {T, D<:AbstractMvUrnDistribution}

	n_groups = length(obs_mean)

	σ² 				~ InverseGamma(1, 1)
	μ_grand 		~ Normal(0, 1)

	# pass the prior like this to the model?
	equal_indices ~ partition_prior

	# The setup for θ follows Rouder et al., 2012, p. 363
	g   ~ InverseGamma(0.5, 0.5)
	θ_r ~ filldist(Normal(0, sqrt(g)), n_groups - 1)
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

@model function one_way_anova_eq_cond_ss(obs_mean, obs_var, obs_n, Q, partition_prior, indicator_state = ones(Int, length(obs_mean)), ::Type{T} = Float64) where {T}

	n_groups = length(obs_mean)

	σ² 				~ InverseGamma(1, 1)
	μ_grand 		~ Normal(0, 1)

	equal_indices = TArray{Int}(n_groups)
	equal_indices .= indicator_state
	for i in eachindex(equal_indices)
		equal_indices[i] ~ get_partition_prior(partition_prior, equal_indices, i)
	end
	indicator_state .= equal_indices

	# TODO add g
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

function fit_model(df::DF.DataFrame; kwargs...) where D<:Union{Nothing, AbstractMvUrnDistribution, AbstractConditionalUrnDistribution}
	obs_mean, obs_var, obs_n = get_suff_stats(df)
	return fit_model(obs_mean, obs_var, obs_n; kwargs...)
end

function fit_model(
			obs_mean::Vector{Float64},
			obs_var::Vector{Float64},
			obs_n::Vector{Int};
			mcmc_iterations::Int = 15_000,
			partition_prior::D = nothing,
			mcmc_burnin::Int = 500,
			use_Gibbs::Bool = true,
			kwargs...
)			where D<:Union{Nothing, AbstractMvUrnDistribution, AbstractConditionalUrnDistribution}
	Q = getQ(length(obs_mean))
	@assert length(obs_mean) == length(obs_var) == length(obs_n)
	if partition_prior !== nothing && length(obs_mean) != length(partition_prior)
		partition_prior = fix_length(partition_prior, length(obs_mean))
	end
	@assert partition_prior === nothing || length(obs_mean) == length(partition_prior)
	model, sampler = get_model_and_sampler(partition_prior, obs_mean, obs_var, obs_n, Q, use_Gibbs)
	return sample_model(model, sampler, mcmc_iterations, mcmc_burnin; kwargs...)
end

function get_model_and_sampler(::Nothing, obs_mean, obs_var, obs_n, Q, ::Bool)
	model   = one_way_anova_full_ss(obs_mean, obs_var, obs_n, Q)
	sampler = NUTS()
	return model, sampler
end

function get_model_and_sampler(partition_prior::AbstractMvUrnDistribution, obs_mean, obs_var, obs_n, Q, use_Gibbs)
	model   = one_way_anova_eq_mv_ss(obs_mean, obs_var, obs_n, Q, partition_prior)
	if use_Gibbs
		sampler = Gibbs(GibbsConditional(:equal_indices, EqualitySampler.PartitionSampler(length(obs_mean), get_log_posterior(obs_mean, obs_var, obs_n, Q, partition_prior))), HMC(0.01, 10, :μ_grand, :σ², :θ_r))
	else
		sampler = Gibbs(PG(20, :equal_indices), HMC(0.01, 10, :μ_grand, :σ², :θ_r))
	end
	return model, sampler
end

function get_model_and_sampler(partition_prior::AbstractConditionalUrnDistribution, obs_mean, obs_var, obs_n, Q, ::Bool)
	model   = one_way_anova_eq_cond_ss(obs_mean, obs_var, obs_n, Q, partition_prior)
	sampler = Gibbs(PG(20, :equal_indices), HMC(0.01, 10, :μ_grand, :σ², :θ_r))
	# sampler = Gibbs(GibbsConditional(:equal_indices, EqualitySampler.PartitionSampler(n_groups, get_log_posterior(obs_mean, obs_var, obs_n, Q, partition_prior))), HMC(0.01, 10, :μ_grand, :σ², :θ_r))
	# sampler = MH()#Gibbs(MH(:equal_indices), HMC(0.01, 10, :μ_grand, :σ², :θ_r))
	return model, sampler
end

function get_log_posterior(obs_mean, obs_var, obs_n, Q, partition_prior::AbstractMvUrnDistribution)
	return function logposterior(nextValues, c)
		σ = sqrt(c.σ²)
		θ_s = Q * c.θ_r
		θ_cs = average_equality_constraints(θ_s, nextValues)
		return sum(logpdf(NormalSuffStat(obs_var[j], c.μ_grand + θ_cs[j], σ, obs_n[j]), obs_mean[j]) for j in eachindex(obs_mean)) + logpdf(partition_prior, nextValues)
	end
end



get_partition_prior(x) = error("use UniformConditionalUrnDistribution or BetaBinomialConditionalUrnDistribution, or implement a custom conditional urn distribution.")
get_partition_prior(::UniformConditionalUrnDistribution, urns, i) = UniformConditionalUrnDistribution(urns, i)
get_partition_prior(D::BetaBinomialConditionalUrnDistribution, urns, i) = BetaBinomialConditionalUrnDistribution(urns, i, D.α, D.β)

fix_length(::UniformMvUrnDistribution, len::Int)		= UniformMvUrnDistribution(len)
fix_length(D::BetaBinomialMvUrnDistribution, len::Int)	= BetaBinomialMvUrnDistribution(len, D.α, D.β)
fix_length(D::RandomProcessMvUrnDistribution, len::Int)	= RandomProcessMvUrnDistribution(len, D.rpm)

function sample_model(model, sampler, iterations, burnin; kwargs...)
	chain = sample(model, sampler, iterations, discard_initial = burnin; kwargs...)
	θ_cs = get_θ_cs(model, chain)
	mean_θ_cs = mean(θ_cs, dims = 2)
	return mean_θ_cs, θ_cs, chain, model
end


function sample_true_model(no_params, no_inequalities)

	target_inequalities = round(Int, (no_params * no_inequalities) / 100)
	d = BetaBinomialMvUrnDistribution(no_params)

	# tweak the inclusion probabilities
	d._log_model_probs_by_incl .= [i == target_inequalities + 1 ? 0.0 : -Inf for i in eachindex(d._log_model_probs_by_incl)]
	u = rand(d)

	@assert count_equalities(u) == target_inequalities

	return u

end

function get_θ(offset, true_model)
	θ = true_model .* offset
	return θ .- mean(θ)
end
