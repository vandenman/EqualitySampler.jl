#=

	This file seems to work as intended!

=#

using Turing, StatsBase, DynamicPPL, FillArrays, Plots
include("src/helperfunctions.jl")

@model function model(x, ::Type{T} = Float64) where {T}

	# sampling equalities only works for k = 3 in this example
	n, k = size(x)

	τ ~ InverseGamma(1, 1)
	ρ ~ Dirichlet(ones(k))
	μ ~ filldist(Normal(0, 5), k)
	# gammas ~ filldist(Gamma(1, 1), k) # assumes alpha = 1
	# ρ = gammas ./ sum(gammas)

	# sample equalities among the sds
	equal_indices = ones(Int, k)
	equal_indices[1] ~ Categorical([1])
	equal_indices[2] ~ Categorical([2/5, 3/5])
	equal_indices[3] ~ Categorical(equal_indices[2] == 1 ? [.5, 0, .5] : 1/3 .* ones(3))

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

# simulate data -- no equalities
n = 100
k = 3
sds = Float64[1, 3, 5]
D = MvNormal(sds)
x = permutedims(rand(D, n))

# fit model with equalities
mod_eq = model(x)
spl_eq = Gibbs(PG(20, :equal_indices), HMC(0.02, 10, :τ, :ρ, :μ))
chn_eq = sample(mod_eq, spl_eq, 5_000);

plottrace(mod_eq, chn_eq)

# examine results for parameters
get_posterior_means_mu_sigma(mod_eq, chn_eq)
plotresults(mod_eq, chn_eq, D)
plotresults(mod_eq, chn_eq, x)

compute_model_counts(chn_eq)
compute_post_prob_eq(chn_eq)



# simulate data -- all equal
n = 100
k = 3
sds = Float64[1, 1, 1]
D = MvNormal(sds)
x = permutedims(rand(D, n))

# fit model with equalities
mod_eq = model(x)
spl_eq = Gibbs(PG(20, :equal_indices), HMC(0.02, 10, :τ, :ρ, :μ))
chn_eq = sample(mod_eq, spl_eq, 5_000);

plottrace(mod_eq, chn_eq)

# examine results for parameters
get_posterior_means_mu_sigma(mod_eq, chn_eq)
plotresults(mod_eq, chn_eq, D)
plotresults(mod_eq, chn_eq, x)

compute_model_counts(chn_eq)
compute_post_prob_eq(chn_eq)
