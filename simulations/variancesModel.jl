using EqualitySampler, Turing, StatsBase, DynamicPPL, FillArrays, Plots
include("simulations/plotFunctions.jl")
include("simulations/helpersTuring.jl")
# include("src/newApproach4.jl")

function apply_equality_constraints(ρ::AbstractVector{<:Real}, equal_indices::AbstractVector{<:Integer})
	ρ_c = similar(ρ)
	for i in eachindex(ρ)
		# two sds are equal if equal_indices[i] == equal_indices[j]
		# if they're equal, ρ2[i] is the average of all ρ[j] ∀ j ∈ 1:k such that equal_indices[i] == equal_indices[j]
		ρ_c[i] = mean(ρ[equal_indices .== equal_indices[i]])
	end
	return ρ_c
end

@model function model_raw(x, uniform = true, indicator_state = ones(Int, size(x)[1]), ::Type{T} = Float64) where {T}

	k, n = size(x)
	# sample equalities among the sds
	equal_indices = TArray{Int}(k)
	equal_indices .= indicator_state
	for i in eachindex(equal_indices)
		if uniform
			equal_indices[i] ~ UniformConditionalUrnDistribution(equal_indices, i)
		else
			equal_indices[i] ~ BetaBinomialConditionalUrnDistribution(equal_indices, i)
		end
	end
	indicator_state .= equal_indices

	τ ~ InverseGamma(1, 1)
	ρ ~ Dirichlet(ones(k))
	μ ~ filldist(Normal(0, 5), k)

	# reassign rho to conform to the sampled equalities
	ρ_c = apply_equality_constraints(ρ, equal_indices)
	# k * τ .* ρ2 gives the precisions of each group
	σ_c = 1 ./ sqrt.(k * τ .* ρ_c)

	for i in axes(x, 2)
		x[:, i] ~ MvNormal(μ, σ_c)
	end
	return (σ_c, ρ_c)
end

@model function model_suffstat(obs_mean, obs_var, n, k, uniform = true, indicator_state = ones(Int, k), ::Type{T} = Float64) where {T}

	# sample equalities among the sds
	equal_indices = TArray{Int}(k)
	equal_indices .= indicator_state
	for i in eachindex(equal_indices)
		if uniform
			equal_indices[i] ~ UniformConditionalUrnDistribution(equal_indices, i)
		else
			equal_indices[i] ~ BetaBinomialConditionalUrnDistribution(equal_indices, i)
		end
	end
	indicator_state .= equal_indices

	τ ~ InverseGamma(1, 1)
	ρ ~ Dirichlet(ones(k))
	μ ~ filldist(Normal(0, 5), k)

	# reassign rho to conform to the sampled equalities
	ρ_c = apply_equality_constraints(ρ, equal_indices)
	# k * τ .* ρ2 gives the precisions of each group
	σ_c = 1 ./ sqrt.(k * τ .* ρ_c)

	obs_mean ~ MvNormalSuffStat(obs_var, μ, σ_c, n)
	return (σ_c, ρ_c)
end

n = 500
k = 5

precs = collect(1.0 : 2 : 2k)
precs[2] = precs[1] = 1
precs[3] = precs[5] = .25
precs[4] = 100
sds = 1 ./ sqrt.(precs)
D = MvNormal(sds)
x = rand(D, n)

uniform_prior = true
mod_eq_raw = model_raw(x, uniform_prior)
prior_distribution = uniform_prior ? UniformConditionalUrnDistribution(ones(Int, k), 1) : BetaBinomialConditionalUrnDistribution(ones(Int, k), 1)

# study prior
chn_eq_raw_prior = sample(mod_eq_raw, Prior(), 20_000)
visualize_helper(prior_distribution, chn_eq_raw_prior)

# study posterior
spl_eq_raw = Gibbs(PG(20, :equal_indices), HMC(0.005, 10, :τ, :ρ, :μ))
chn_eq_raw = sample(mod_eq_raw, spl_eq_raw, 5_000)
visualize_helper(prior_distribution, chn_eq_raw)

plottrace(mod_eq_raw, chn_eq_raw)
get_posterior_means_mu_sigma(mod_eq_raw, chn_eq_raw)
plotresults(mod_eq_raw, chn_eq_raw, D)
plotresults(mod_eq_raw, chn_eq_raw, permutedims(x))


# sufficient statistics
obsmu  = vec(mean(x, dims = 2))
obsmu2 = vec(mean(x->x^2, x, dims = 2))

@assert loglikelihood(D, x) ≈ loglikelihood(MvNormalSuffStat(obsmu2, mean(D), sqrt.(var(D)), n), obsmu)


uniform_prior = false
mod_eq_ss = model_suffstat(obsmu, obsmu2, n, k, uniform_prior)
prior_distribution = uniform_prior ? UniformConditionalUrnDistribution(ones(Int, k), 1) : BetaBinomialConditionalUrnDistribution(ones(Int, k), 1)

# study prior
chn_eq_ss_prior = sample(mod_eq_ss, Prior(), 20_000)
visualize_helper(prior_distribution, chn_eq_ss_prior)
# compute_post_prob_eq(chn_eq_ss_prior)
# empirical_model_probs = compute_model_probs(chn_eq_ss_prior)
# empirical_inclusion_probs = compute_incl_probs(chn_eq_ss_prior)
# visualize_eq_samples(prior_distribution, empirical_model_probs, empirical_inclusion_probs)

# study posterior
# spl_eq_ss = Gibbs(PG(20, :equal_indices), HMC(0.01, 10, :τ, :ρ, :μ))
spl_eq_ss = Gibbs(PG(20, :equal_indices), HMC(0.001, 10, :τ, :ρ, :μ))
chn_eq_ss = sample(mod_eq_ss, spl_eq_ss, 25_000)
visualize_helper(prior_distribution, chn_eq_ss)

plottrace(mod_eq_ss, chn_eq_ss)
get_posterior_means_mu_sigma(mod_eq_ss, chn_eq_ss)
plotresults(mod_eq_ss, chn_eq_ss, D)
plotresults(mod_eq_ss, chn_eq_ss, permutedims(x))

post_model_probs = sort(compute_model_probs(chn_eq_ss), byvalue=true)
df = DataFrame(
	model = collect(keys(post_model_probs)),
	postprob = collect(values(post_model_probs)),
	no_incl = count_equalities.(collect(keys(post_model_probs)))
)
show(df, allrows = true)

sqrt.(var(D))
display(LinearAlgebra.UnitLowerTriangular(compute_post_prob_eq(chn_eq_ss)))


