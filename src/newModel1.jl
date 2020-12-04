using Turing, StatsBase, DynamicPPL, FillArrays, Plots
include("src/loglikelihood.jl")
include("src/helperfunctions.jl")
include("src/newApproach2.jl")

@model function model(x, ::Type{T} = Float64) where {T}

	n, k = size(x)

	τ ~ InverseGamma(1, 1)
	ρ ~ Dirichlet(ones(k))
	μ ~ filldist(Normal(0, 5), k)

	# sample equalities among the sds
	equal_indices = Vector{Int}(undef, k)
	for i in eachindex(equal_indices)
		equal_indices[i] ~ DiscreteUniform(1, 4)
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

@model function model_suffstat(obs_mean, obs_var, n, k, ::Type{T} = Float64) where {T}

	τ ~ InverseGamma(1, 1)
	ρ ~ Dirichlet(ones(k)) # should this be adjusted by equal_indices?
	μ ~ filldist(Normal(0, 5), k)

	# sample equalities among the sds
	equal_indices = Vector{Int}(undef, k)
	for i in eachindex(equal_indices)
		equal_indices[i] ~ DiscreteUniform(1, 4)
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

	Turing.@addlogprob! multivariate_normal_likelihood(obs_mean, obs_var, μ, σ_c, n)
	return (σ_c, ρ_c)
end

n = 100
k = 3

sds = collect(1.0 : 2 : 2k) # 1, 3, 5, ...
# sds = 1 ./ sqrt.(precs)
D = MvNormal(sds)
x = permutedims(rand(D, n))

# fit model -- MvNormal
mod_eq = model(x)
spl_eq = Gibbs(PG(20, :equal_indices), HMC(0.01, 10, :τ, :ρ, :μ))
# spl_eq = Gibbs(PG(20, :equal_indices), NUTS())
chn_eq = sample(mod_eq, spl_eq, 2_000, n_adapts = 1_000, drop_warmup = true);

plottrace(mod_eq, chn_eq)

# examine results for parameters
get_posterior_means_mu_sigma(mod_eq, chn_eq)
plotresults(mod_eq, chn_eq, D)
plotresults(mod_eq, chn_eq, x)

compute_post_prob_eq(chn_eq)
compute_model_counts(chn_eq)
model_probs_eq = compute_model_probs(chn_eq)
sort(collect(model_probs_eq), by=x->x[2])

# fit model -- fast logposterior
obsmu  = vec(mean(x, dims = 1))
obsmu2 = vec(mean(x->x^2, x, dims = 1))
mod_eq_ss = model_suffstat(obsmu, obsmu2, n, k)
spl_eq_ss = Gibbs(PG(20, :equal_indices), HMC(0.02, 10, :τ, :ρ, :μ))
chn_eq_ss = sample(mod_eq_ss, spl_eq_ss, 5_000, n_adapts = 2_000, drop_warmup = true);

plottrace(mod_eq_ss, chn_eq_ss)

# examine results for parameters
get_posterior_means_mu_sigma(mod_eq_ss, chn_eq_ss)
plotresults(mod_eq_ss, chn_eq_ss, D)
plotresults(mod_eq_ss, chn_eq_ss, x)

compute_post_prob_eq(chn_eq_ss)
compute_model_counts(chn_eq_ss)
model_probs_eq_ss = compute_model_probs(chn_eq_ss)
sort(collect(model_probs_eq_ss), by=x->x[2])
