import	StatsBase 			as SB,
		LinearAlgebra 		as LA,
		StatsModels			as SM,
		DataFrames			as DF,
		GLM

import Statistics: mean, var
# import Suppressor
import CategoricalArrays
using Turing
import Memoize

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

	length(θ) != n_groups && throw(error("length(θ) != n_groups"))

	n_obs = n_groups * n_obs_per_group
	θ2 = θ .- SB.mean(θ)

	g = Vector{Int}(undef, n_obs)
	for (i, r) in enumerate(Iterators.partition(1:n_obs, ceil(Int, n_obs / n_groups)))
		g[r] .= i
	end

	D = MvNormal(μ .+ σ .* θ2[g], σ)
	y = rand(D)

	df = DF.DataFrame(:y => y, :g => CategoricalArrays.categorical(g))

	true_values = Dict{Symbol, Union{Float64, Vector{Float64}, Vector{Int}}}(
		:σ					=> σ,
		:μ					=> μ,
		:θ					=> θ2,
		:true_partition		=> true_partition
	)

	return y, df, D, true_values

end

fit_lm(y, df) = fit_lm(df)
function fit_lm(df)
	mod = SM.fit(GLM.LinearModel, SM.@formula(y ~ 1 + g), df)#, contrasts = Dict(:g => StatsModels.FullDummyCoding()))
	# transform the coefficients to a grand mean and offsets
	coefs = copy(GLM.coef(mod))
	ests = similar(coefs)
	ests[1] = coefs[1] - mean(df[!, :y])
	ests[2:end] .= coefs[2:end] .+ coefs[1] .- mean(df[!, :y])
	return ests, mod
end

getQ(n_groups::Integer)::Matrix{Float64} = getQ_Stan(n_groups)

function getQ_Rouder(n_groups::Integer)::Matrix{Float64}
	# X = StatsModels.modelmatrix(@formula(y ~ 0 + g).rhs, DataFrame(:g => g), hints = Dict(:g => StatsModels.FullDummyCoding()))
	Σₐ = Matrix{Float64}(LA.I, n_groups, n_groups) .- (1.0 / n_groups)
	_, v::Matrix{Float64} = LA.eigen(Σₐ)
	Q = v[end:-1:1, end:-1:2] # this is what happens in Rouder et al., (2012) eq ...

	@assert isapprox(sum(Q * randn(n_groups-1)), 0.0, atol = 1e-8)

	return Q
end

function getQ_Stan(K::Integer)::Matrix{Float64}
	# Stan approach: https://mc-stan.org/docs/2_18/stan-users-guide/parameterizing-centered-vectors.html
	A = Matrix(LA.Diagonal(ones(K)))
	for i in 1:K-1
		A[K, i] = -1;
	end
	A[K,K] = 0;
	return LA.qr(A).Q[:, 1:K-1]
end

function getQ_Stan_2(K::Integer)::Vector{Float64}
	# Stan approach: https://mc-stan.org/docs/2_18/stan-users-guide/parameterizing-centered-vectors.html
	Q_r = Vector{Float64}(undef, 2K)
	for i in 1:K
		Q_r[i]   = -sqrt((K - i) / (K - i + 1.0))
		Q_r[i+K] = inv(sqrt((K - i) * (K - i + 1)))
	end
	return Q_r
end

function apply_Q_Stan_2(x_raw::Vector{T}, Q_r::Vector{Float64}) where T
	K = length(x_raw) + 1
	x = Vector{T}(undef, K)
	x_aux = zero(T)

	for i in 1:K-1
	  x[i]  = x_aux + x_raw[i] * Q_r[i]
	  x_aux = x_aux + x_raw[i] * Q_r[i+K]
	end

	x[K] = x_aux
	return x
end

@model function one_way_anova_full_ss(obs_mean, obs_var, obs_n, Q, ::Type{T} = Float64) where {T}

	n_groups = length(obs_mean)
	σ² ~ InverseGamma(1, 1)
	μ_grand ~ Normal(0, 1)

	# The setup for θ follows Rouder et al., 2012, p. 363
	θ_r ~ MvNormal(n_groups - 1, 1.0)#, n_groups - 1)
	# θ_r ~ filldist(Normal(0, 1), n_groups - 1)
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

	temp_df::DF.DataFrame = DF.combine(DF.groupby(df, :g), :y => mean, :y => var, :y => length)

	obs_mean::Vector{Float64}	= temp_df[!, "y_mean"]
	obs_var::Vector{Float64}	= temp_df[!, "y_var"]
	obs_n::Vector{Int}			= temp_df[!, "y_length"]

	return obs_mean, obs_var, obs_n
end

function get_θ_cs(model, chain)
	gen = generated_quantities2(model, chain)
	θ_cs = Matrix{Float64}(undef, length(first(gen[1])), length(gen))
	for i in eachindex(gen)
		θ_cs[:, i] = gen[i][1]
	end
	return θ_cs
end

function average_equality_constraints!(ρ::AbstractVector{<:Real}, partition::AbstractVector{<:Integer})
	idx_vecs = [Int[] for _ in eachindex(partition)]
	@inbounds for i in eachindex(partition)
		push!(idx_vecs[partition[i]], i)
	end

	@inbounds for idx in idx_vecs
		isempty(idx) && continue
		ρ[idx] .= mean(ρ[idx])
	end
	return ρ
end

average_equality_constraints(ρ::AbstractVector{<:Real}, partition::AbstractVector{<:Integer}) = average_equality_constraints!(copy(ρ), partition)


# Memoize.@memoize function get_idx_vecs(partition::AbstractVector{<:Integer})
# 	idx_vecs = Dict{Int, Vector{Int}}()
# 	@inbounds for i in eachindex(partition)
# 		key = @inbounds partition[i]
# 		if haskey(idx_vecs, key)
# 			push!(idx_vecs[key], i)
# 		else
# 			idx_vecs[key] = [i]
# 		end
# 	end
# 	idx_vecs
# end

# function average_equality_constraints_memoized(ρ::AbstractVector{<:Real}, partition::AbstractVector{<:Integer})

# 	idx_vecs = get_idx_vecs(partition)

# 	ρ_c = similar(ρ)
# 	@inbounds for idx in values(idx_vecs)
# 		ρ_c[idx] .= mean(ρ[idx])
# 	end

# 	return ρ_c

# end

@model function one_way_anova_eq_mv_ss(obs_mean, obs_var, obs_n, Q, partition_prior::D, ::Type{T} = Float64) where {T, D<:AbstractMvUrnDistribution}

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

@model function one_way_anova_eq_cond_ss(obs_mean, obs_var, obs_n, Q, partition_prior, indicator_state = ones(Int, length(obs_mean)), ::Type{T} = Float64) where {T}

	n_groups = length(obs_mean)

	σ² 				~ InverseGamma(1, 1)
	μ_grand 		~ Normal(0, 1)

	partition = TArray{Int}(n_groups)
	partition .= indicator_state
	for i in eachindex(partition)
		partition[i] ~ get_partition_prior(partition_prior, partition, i)
	end
	indicator_state .= partition

	# TODO add g
	# The setup for θ follows Rouder et al., 2012, p. 363
	θ_r ~ filldist(Normal(0, 1), n_groups - 1)
	# ensure that subtract the mean of θ to ensure the sum to zero constraint
	θ_s = Q * θ_r
	# constrain θ according to the sampled equalities
	θ_cs = average_equality_constraints(θ_s, partition)

	# definition from Rouder et. al., (2012) eq 6.
	for i in 1:n_groups
		obs_mean[i] ~ NormalSuffStat(obs_var[i], μ_grand + sqrt(σ²) * θ_cs[i], σ², obs_n[i])
	end
	return (θ_cs, )
end

function fit_model(df::DF.DataFrame; kwargs...)
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
			hmc_stepsize::Float64 = 0.0,
			n_leapfrog::Int = 10,
			model = nothing,
			custom_hmc_adaptation::Bool = true,
			custom_hmc = nothing,
			kwargs...
)			where D<:Union{Nothing, AbstractMvUrnDistribution, AbstractConditionalUrnDistribution}
	Q = getQ(length(obs_mean))
	@assert length(obs_mean) == length(obs_var) == length(obs_n)
	if partition_prior !== nothing && length(obs_mean) != length(partition_prior)
		partition_prior = fix_length(partition_prior, length(obs_mean))
	end
	@assert partition_prior === nothing || length(obs_mean) == length(partition_prior)
	model, sampler = get_model_and_sampler(partition_prior, obs_mean, obs_var, obs_n, Q, use_Gibbs; hmc_stepsize = hmc_stepsize, n_leapfrog = n_leapfrog, model, custom_hmc)
	init_theta = get_initial_values(model, obs_mean, obs_var, obs_n, Q, partition_prior)
	if custom_hmc_adaptation && isnothing(custom_hmc)
		sampler = sampler_with_custom_adaptation(model, sampler, init_theta, partition_prior, obs_mean, obs_var, obs_n, Q, n_leapfrog)
	end
	return sample_model(model, sampler, mcmc_iterations, mcmc_burnin, init_theta; kwargs...)
end

function get_model_and_sampler(::Nothing, obs_mean, obs_var, obs_n, Q, ::Bool)
	model   = one_way_anova_full_ss(obs_mean, obs_var, obs_n, Q)
	sampler = NUTS()
	return model, sampler
end

function get_model_and_sampler(partition_prior::AbstractMvUrnDistribution, obs_mean, obs_var, obs_n, Q, use_Gibbs; hmc_stepsize = 0.0, n_leapfrog::Int = 10, model, custom_hmc)
	if isnothing(model)
		model = one_way_anova_eq_mv_ss(obs_mean, obs_var, obs_n, Q, partition_prior)
	elseif model isa Function
		model = model(obs_mean, obs_var, obs_n, Q, partition_prior)
	elseif model isa DynamicPPL.Model
		# nothing
	else
		throw(DomainError(model, "Model should be nothing, a function, or a DynamicPPL.Model"))
	end

	# @show hmc_stepsize, n_leapfrog
	if use_Gibbs
		sampler = Gibbs(
			GibbsConditional(:partition, EqualitySampler.PartitionSampler(length(obs_mean), get_log_posterior(obs_mean, obs_var, obs_n, Q, partition_prior))),
			isnothing(custom_hmc) ? HMC(hmc_stepsize, n_leapfrog, :μ_grand, :σ², :θ_r, :g) : custom_hmc
			# HMC(hmc_stepsize, n_leapfrog, :μ_grand),
			# HMC(hmc_stepsize, n_leapfrog, :σ²),
			# HMC(hmc_stepsize, n_leapfrog, :θ_r),
			# HMC(hmc_stepsize, n_leapfrog, :g)
			# MH(
			# 	:μ_grand	=> AdvancedMH.RandomWalkProposal(Normal(0.0, 0.1)),
			# 	:θ_r		=> AdvancedMH.RandomWalkProposal(MvNormal(length(obs_mean) - 1, 0.1)),
			# 	:σ²			=> AdvancedMH.RandomWalkProposal(Truncated(Normal(0, 0.1), 0.0, Inf)),
			# 	:g			=> AdvancedMH.RandomWalkProposal(Truncated(Normal(0, 0.1), 0.0, Inf))
			# )
		)
	else
		sampler = Gibbs(
			PG(20, :partition),
			isnothing(custom_hmc) ? HMC(hmc_stepsize, n_leapfrog, :μ_grand, :σ², :θ_r, :g) : custom_hmc
		)
	end

	return model, sampler
end

function sampler_with_custom_adaptation(model, spl0, init_theta, partition_prior, obs_mean, obs_var, obs_n, Q, n_leapfrog)
	hmc_stepsize = custom_hmc_adaptation(model, spl0, init_theta)
	return Gibbs(
		GibbsConditional(:partition, EqualitySampler.PartitionSampler(length(obs_mean), get_log_posterior(obs_mean, obs_var, obs_n, Q, partition_prior))),
		HMC(hmc_stepsize, n_leapfrog, :μ_grand, :σ², :θ_r, :g)
	)
end

function get_model_and_sampler(partition_prior::AbstractConditionalUrnDistribution, obs_mean, obs_var, obs_n, Q, ::Bool)
	model   = one_way_anova_eq_cond_ss(obs_mean, obs_var, obs_n, Q, partition_prior)
	sampler = Gibbs(PG(20, :partition), HMC(0.01, 10, :μ_grand, :σ², :θ_r))
	# sampler = Gibbs(GibbsConditional(:partition, EqualitySampler.PartitionSampler(n_groups, get_log_posterior(obs_mean, obs_var, obs_n, Q, partition_prior))), HMC(0.01, 10, :μ_grand, :σ², :θ_r))
	# sampler = MH()
	# sampler = Gibbs(MH(:partition), HMC(0.01, 10, :μ_grand, :σ², :θ_r))
	return model, sampler
end

function get_log_posterior(obs_mean, obs_var, obs_n, Q, partition_prior::AbstractMvUrnDistribution)
	return function logposterior(nextValues, c)
		# σ = sqrt(c.σ²)
		θ_s = Q * c.θ_r
		θ_cs = average_equality_constraints(θ_s, nextValues)
		return sum(logpdf(NormalSuffStat(obs_var[j], c.μ_grand + sqrt(c.σ²) * θ_cs[j], c.σ², obs_n[j]), obs_mean[j]) for j in eachindex(obs_mean)) + logpdf(partition_prior, nextValues)
	end
end

get_partition_prior(x) = error("use UniformConditionalUrnDistribution or BetaBinomialConditionalUrnDistribution, or implement a custom conditional urn distribution.")
get_partition_prior(::UniformConditionalUrnDistribution, urns, i) = UniformConditionalUrnDistribution(urns, i)
get_partition_prior(D::BetaBinomialConditionalUrnDistribution, urns, i) = BetaBinomialConditionalUrnDistribution(urns, i, D.α, D.β)

fix_length(::UniformMvUrnDistribution, len::Int)		= UniformMvUrnDistribution(len)
fix_length(D::BetaBinomialMvUrnDistribution, len::Int)	= BetaBinomialMvUrnDistribution(len, D.α, D.β)
fix_length(D::RandomProcessMvUrnDistribution, len::Int)	= RandomProcessMvUrnDistribution(len, D.rpm)

function sample_model(model, sampler, iterations, burnin, init_theta; kwargs...)
	chain = sample(model, sampler, iterations, discard_initial = burnin, init_theta = init_theta; kwargs...)
	θ_cs = get_θ_cs(model, chain)
	mean_θ_cs = mean(θ_cs, dims = 2)
	return mean_θ_cs, θ_cs, chain, model, sampler
end

# function sample_true_model(no_params, no_inequalities)

# 	target_inequalities = round(Int, (no_params * no_inequalities) / 100)
# 	d = BetaBinomialMvUrnDistribution(no_params)

# 	# tweak the inclusion probabilities
# 	d._log_model_probs_by_incl .= [i == target_inequalities + 1 ? 0.0 : -Inf for i in eachindex(d._log_model_probs_by_incl)]
# 	u = rand(d)

# 	@assert count_equalities(u) == target_inequalities

# 	return u

# end

function sample_true_model(no_params, no_inequalities)

	target_inequalities = round(Int, ((no_params - 1)  * no_inequalities) / 100)
	d = BetaBinomialMvUrnDistribution(no_params)

	# tweak the inclusion probabilities
	d._log_model_probs_by_incl .= [i == target_inequalities + 1 ? 0.0 : -Inf for i in eachindex(d._log_model_probs_by_incl)]
	u = rand(d)

	@assert count_equalities(u) == target_inequalities

	return u

end


function get_θ(offset, true_model::Vector{T}) where T<:Integer

	copy_model = copy(true_model)
	current_max = copy_model[1]

	for i in eachindex(copy_model)
		if copy_model[i] > current_max
			copy_model[copy_model .== i] .= current_max
			current_max += 1
		elseif copy_model[i] == i
			current_max += 1
		end
	end

	θ = copy_model .* offset
	return θ .- mean(θ)
end

compute_retrieval(true_model::Vector{Int}, estimated_model::Vector{Int}) = compute_retrieval(BitMatrix(i == j for i in true_model, j in true_model), BitMatrix(i == j for i in estimated_model, j in estimated_model))
compute_retrieval(true_model::Vector{Int}, estimated_model::BitMatrix) = compute_retrieval(BitMatrix(i == j for i in true_model, j in true_model), estimated_model)
function compute_retrieval(true_model::BitMatrix, estimated_model::BitMatrix)

	# @show true_model, estimated_model
	@assert size(true_model) == size(estimated_model)

	false_equalities		= 0
	false_inequalities		= 0 # <- examine this to control alpha
	true_equalities			= 0
	true_inequalities		= 0

	no_params = length(true_model)
	for j in 1:size(true_model, 1)-1, i in j+1:size(true_model, 1)
		if true_model[i, j]
			if estimated_model[i, j]
				true_equalities += 1
			else
				false_inequalities += 1
			end
		else
			if estimated_model[i, j]
				false_equalities += 1
			else
				true_inequalities += 1
			end
		end
	end

	return NamedTuple{(:false_equalities, :false_inequalities, :true_equalities, :true_inequalities), NTuple{4,Int}}((
		false_equalities, false_inequalities, true_equalities, true_inequalities
	))

	# return (false_equalities=false_equalities, false_inequalities=false_inequalities, true_equalities=true_equalities, true_inequalities=true_inequalities)

end

function incl_probs_to_model(included)

	no_params = size(included, 1)
	estimated_model = Vector{Int}(1:no_params)

	for i in 1:no_params-1, j in i+1:no_params
		if included[j, i]
			estimated_model[j] = estimated_model[i]
		end
	end

	return estimated_model
end

function get_initial_values(model, obs_mean, obs_var, obs_n, Q, partition_prior)

	# weighted mean as grand mean
	μ_grand_init = LA.dot(obs_mean, obs_n) / sum(obs_n)
	# pooled variance
	σ²_init = sum((obs_n[i] - 1) * obs_var[i] for i in eachindex(obs_var)) / (sum(obs_n) - length(obs_n))
	θ_r_init = LA.pinv(Q) * obs_mean
	# this assert is a bit wonky (also with isapprox)
	# @assert all(x->abs(x) <= 1e-1 , obs_mean .- Q * θ_r_init)

	varinfo = Turing.VarInfo(model);
	if partition_prior === nothing
		priorcontextargs = (
			μ_grand			= μ_grand_init,
			σ²				= σ²_init,
			θ_r				= sqrt(σ²_init) .* θ_r_init,
			g				= std(θ_r_init)
		)
	else
		priorcontextargs = (
			μ_grand			= μ_grand_init,
			σ²				= σ²_init,
			θ_r				= sqrt(σ²_init) .* θ_r_init,
			partition		= collect(eachindex(obs_mean)),
			g				= isone(length(θ_r_init)) ? 1.0 : std(θ_r_init)
		)
	end

	model(varinfo, Turing.SampleFromPrior(), Turing.PriorContext(priorcontextargs))
	init_theta = varinfo[Turing.SampleFromPrior()]
	return init_theta
end


