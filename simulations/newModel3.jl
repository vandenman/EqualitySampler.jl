
using Turing, StatsBase, DynamicPPL, FillArrays, Plots
include("src/loglikelihood.jl")
include("src/helperfunctions.jl")
include("src/newApproach3.jl")

function visualize_eq_samples(equalityPrior, empirical_model_probs, empirical_inclusion_probs)
	p1 = plot_modelspace(equalityPrior, empirical_model_probs);
	p2 = plot_inclusionprobabilities(equalityPrior, empirical_inclusion_probs);
	p3 = plot_expected_vs_empirical(equalityPrior, empirical_model_probs);
	pjoint = plot(p1, p2, p3, layout = grid(3, 1), title = ["log model probabilities" "log inclusion probability" "expected vs empirical log model probability"],
				  size = (600, 1200))
end

# @model function priorOnlyUniform()
# 	equal_indices = TArray{Int}(k)
# 	equal_indices .= rand(1:k, k)
# 	for i in eachindex(equal_indices)
# 		equal_indices[i] ~ UniformConditionalUrnDistribution(equal_indices, i)
# 	end
# end

# @model function priorOnlyBetaBinom()
# 	equal_indices = TArray{Int}(k)
# 	equal_indices .= rand(1:k, k)
# 	for i in eachindex(equal_indices)
# 		equal_indices[i] ~ UniformConditionalUrnDistribution(equal_indices, i)
# 	end
# end


# # fit model -- MvNormal
# mod_priorOnly = priorOnly()
# chn_priorOnly = sample(mod_priorOnly, Prior(), 20_000, n_adapts = 1_000, drop_warmup = true);
# compute_post_prob_eq(chn_priorOnly)
# compute_model_probs(chn_priorOnly)

@model function mvmodel(x, uniform = true, ::Type{T} = Float64) where {T}

	n, k = size(x)

	τ ~ InverseGamma(1, 1)
	ρ ~ Dirichlet(ones(k))
	μ ~ filldist(Normal(0, 5), k)

	# sample equalities among the sds
	equal_indices ~ UniformMvUrnDistribution(k)

	# Method I: reassign rho to conform to the sampled equalities
	ρ_c = Vector{T}(undef, k)
	for i in 1:k
		# two sds are equal if equal_indices[i] == equal_indices[j]
		# if they're equal, ρ2[i] is the average of all ρ[j] ∀ j ∈ 1:k such that equal_indices[i] == equal_indices[j]
		ρ_c[i] = mean(ρ[equal_indices .== equal_indices[i]])
	end
	# k * τ .* ρ2 gives the precisions of each group
	σ_c = 1 ./ sqrt.(k * τ .* ρ_c)

	# Method II: use equal_indices as indicator
	# σ0 = 1 ./ sqrt.(k * τ .* ρ)
	# ρ_c = ρ[equal_indices]
	# σ_c = σ0[equal_indices]

	for i in axes(x, 1)
		x[i, :] ~ MvNormal(μ, σ_c)
	end
	return (σ_c, ρ_c)
end

@model function model(x, uniform = true, ::Type{T} = Float64) where {T}

	n, k = size(x)

	τ ~ InverseGamma(1, 1)
	ρ ~ Dirichlet(ones(k))
	μ ~ filldist(Normal(0, 5), k)

	# sample equalities among the sds
	equal_indices = TArray{Int}(k)
	equal_indices .= 1
	for i in eachindex(equal_indices)
		if uniform
			equal_indices[i] ~ UniformConditionalUrnDistribution(equal_indices, i)
		else
			equal_indices[i] ~ BetaBinomialConditionalUrnDistribution(equal_indices, i)
		end
	end

	# Method I: reassign rho to conform to the sampled equalities
	ρ_c = Vector{T}(undef, k)
	for i in 1:k
		# two sds are equal if equal_indices[i] == equal_indices[j]
		# if they're equal, ρ2[i] is the average of all ρ[j] ∀ j ∈ 1:k such that equal_indices[i] == equal_indices[j]
		ρ_c[i] = mean(ρ[equal_indices .== equal_indices[i]])
	end
	# k * τ .* ρ2 gives the precisions of each group
	σ_c = 1 ./ sqrt.(k * τ .* ρ_c)

	# Method II: use equal_indices as indicator
	# σ0 = 1 ./ sqrt.(k * τ .* ρ)
	# ρ_c = ρ[equal_indices]
	# σ_c = σ0[equal_indices]

	for i in axes(x, 1)
		x[i, :] ~ MvNormal(μ, σ_c)
	end
	return (σ_c, ρ_c)
end

@model function model_suffstat(obs_mean, obs_var, n, k, uniform = true, ::Type{T} = Float64) where {T}

	τ ~ InverseGamma(1, 1)
	ρ ~ Dirichlet(ones(k)) # should this be adjusted by equal_indices?
	μ ~ filldist(Normal(0, 5), k)

	# sample equalities among the sds
	equal_indices = TArray{Int}(k)
	# equal_indices .= 1
	equal_indices .= rand(1:k, k) # mitigates bias
	for i in eachindex(equal_indices)
		if uniform
			equal_indices[i] ~ UniformConditionalUrnDistribution(equal_indices, i)
		else
			equal_indices[i] ~ BetaBinomialConditionalUrnDistribution(equal_indices, i)
		end
	end

	# Method I: reassign rho to conform to the sampled equalities
	ρ_c = Vector{T}(undef, k)
	for i in 1:k
		# two sds are equal if equal_indices[i] == equal_indices[j]
		# if they're equal, ρ2[i] is the average of all ρ[j] ∀ j ∈ 1:k such that equal_indices[i] == equal_indices[j]
		ρ_c[i] = mean(ρ[equal_indices .== equal_indices[i]])
	end
	# k * τ .* ρ2 gives the precisions of each group
	σ_c = 1 ./ sqrt.(k * τ .* ρ_c)

	# Method II: use equal_indices as indicator
	# σ0 = 1 ./ sqrt.(k * τ .* ρ)
	# ρ_c = ρ[equal_indices]
	# σ_c = σ0[equal_indices]

	obs_mean ~ ssMvNormal(μ, σ_c, obs_var, n)
	# Turing.@addlogprob! multivariate_normal_likelihood(obs_mean, obs_var, μ, σ_c, n)
	return (σ_c, ρ_c)
end

n = 1000
k = 4

sds = collect(1.0 : 2 : 2k) # 1, 3, 5, ...
# sds = 1 ./ sqrt.(precs)
D = MvNormal(sds)
x = permutedims(rand(D, n))

# fit model -- MvNormal
uniform_prior = false
equalityPrior = uniform_prior ? UniformConditionalUrnDistribution(ones(Int, k), 1) : BetaBinomialConditionalUrnDistribution(ones(Int, k), 1)
mod_eq = model(x, uniform_prior)

# study prior
chn_eq_prior = sample(mod_eq, Prior(), 100_000);
empirical_model_probs = compute_model_probs(chn_eq_prior)
empirical_inclusion_probs = compute_incl_probs(chn_eq_prior)
visualize_eq_samples(equalityPrior, empirical_model_probs, empirical_inclusion_probs)

# study posterior
spl_eq = Gibbs(PG(20, :equal_indices), HMC(0.005, 10, :τ, :ρ, :μ))
chn_eq = sample(mod_eq, spl_eq, 2_000, n_adapts = 1_000, drop_warmup = true);

plottrace(mod_eq, chn_eq)
visualize_eq_samples(chn_eq_prior)

# counts equal to x1 == x2
foo(n) = Combinatorics.stirlings1.(n, 0:n) .* binomial.(n, 0:n)

k = 6
foo(k-1)
opts = Iterators.product(vcat([1:1, 1:1], fill(1:k, k-2))...)
res = sort(countmap(vec([count_equalities(collect(it)) for it in opts])))

length(Iterators.product(fill(1:k, k)...))

length(Iterators.product(fill(1:k, k-2)...))