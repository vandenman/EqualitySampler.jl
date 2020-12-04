using Turing, StatsBase, DynamicPPL, FillArrays, Plots
include("src/loglikelihood.jl")
get_eq_ind_nms(samples) = filter(x->startswith(string(x), "equal_indices"), samples.name_map.parameters)

function compute_post_prob_eq(samples)
	# compute the proportion of samples where equal_indices[i] == equal_indices[j] ∀i, j
	eq_ind_nms = get_eq_ind_nms(samples)
	s = size(samples[eq_ind_nms])
	samps = reshape(samples[eq_ind_nms].value.data, :, s[2])
	n_samps, n_groups = size(samps)
	probs = zeros(Float64, n_groups, n_groups)
	for row in eachrow(samps)
		for j in eachindex(row)
			idx = j .+ findall(==(row[j]), row[j+1:end])
			probs[idx, j] .+= 1.0
		end
	end
	return probs ./ n_samps
end

function get_posterior_means_mu_sigma(model, chn)
	s = summarystats(chn)
	μ = s.nt.mean[startswith.(string.(s.nt.parameters), "μ")]
	gen = generated_quantities(model, chn)
	σ = collect(mean(first(x)[j] for x in gen) for j in 1:k)
	return (μ, σ)
end

"""
plot true values vs posterior means
"""
function plotresults(model, chain, D::MvNormal)
	μ, σ = get_posterior_means_mu_sigma(model, chain)
	plotresultshelper(μ, σ, D.μ, sqrt.(D.Σ.diag))
end

"""
plot observed values vs posterior means
"""
function plotresults(model, chain, x::AbstractMatrix)
	μ, σ = get_posterior_means_mu_sigma(model, chain)
	plotresultshelper(μ, σ, mean(x, dims = 2), sqrt.(var(x, dims = 2)))
end

function plotresultshelper(μ, σ, obsμ, obsσ)
	plot_μ = scatter(obsμ, μ, title = "μ", legend = false);
	Plots.abline!(plot_μ, 1, 0);
	plot_σ = scatter(obsσ, σ, title = "σ", legend = false);
	Plots.abline!(plot_σ, 1, 0);
	plot(plot_μ, plot_σ, layout = (2, 1))
end

@model function model(obs_mean, obs_mean_sq, n, sample_equalities, ::Type{T} = Float64) where {T}

	# sampling equalities only works for k = 3 in this example
	k = length(obs_mean)

	τ ~ InverseGamma(1, 1)
	# ρ ~ Dirichlet(ones(k))
	gammas ~ filldist(Gamma(1, 1), k) # assumes alpha = 1
	μ ~ filldist(Normal(0, 10), k)
	ρ = gammas ./ sum(gammas)
	if sample_equalities

		# sample equalities among the sds
		equal_indices = ones(Int, k)
		equal_indices[1] ~ Categorical([1])
		equal_indices[2] ~ Categorical([2/5, 3/5])
		equal_indices[3] ~ Categorical(equal_indices[2] == 1 ? [.5, 0, .5] : 1/3 .* ones(3))

		# Method I: reassign rho to conform to the sampled equalities
		# ρ2 = Vector{T}(undef, k)
		# for i in 1:k
		# 	# two sds are equal if equal_indices[i] == equal_indices[j]
		# 	# if they're equal, ρ2[i] is the average of all ρ[j] ∀ j ∈ 1:k such that equal_indices[i] == equal_indices[j]
		# 	ρ2[i] = mean(ρ[equal_indices .== equal_indices[i]])
		# end
		# k * τ .* ρ2 gives the precisions of each group
		# σ = 1 ./ sqrt.(k * τ .* ρ2)

		# Method II: use equal_indices as indicator
		σ0 = 1 ./ sqrt.(k * τ .* ρ)
		ρ2 = ρ[equal_indices]
		σ = σ0[equal_indices]
	else # do not sample equalities
		# k * τ .* ρ  gives the precisions of each group
		σ = 1 ./ sqrt.(k * τ .* ρ)
	end

	# likelihood
	Turing.@addlogprob! multivariate_normal_likelihood(obs_mean, obs_mean_sq, μ, σ, n)

	if sample_equalities
		return (σ, ρ2)
	else
		return (σ,   )
	end
end

# simulate data
n = 1000
k = 3
sds = Float64[1, 3, 5]
D = MvNormal(sds)
x = rand(D, n)
obsmu  = mean(x, dims = 2)
obsmu2 = mean(x .^ 2, dims = 2)
sqrt.(var(x, dims = 2))

# fit model without equalities as a rationality check
mod_no_eq = model(obsmu, obsmu2, n, false)
spl_no_eq = NUTS()#Gibbs(HMC(0.001, 10, :τ, :ρ, :μ))
chn_no_eq = sample(mod_no_eq, spl_no_eq, 5_000);

# examine results
get_posterior_means_mu_sigma(mod_no_eq, chn_no_eq)
plotresults(mod_no_eq, chn_no_eq, D)
plotresults(mod_no_eq, chn_no_eq, x)

# fit model with equalities
mod_eq = model(obsmu, obsmu2, n, true)
# spl_eq = Gibbs(PG(20, :equal_indices), HMC(0.003125, 5, :τ, :ρ, :μ))
spl_eq = Gibbs(PG(20, :equal_indices), HMC(0.004, 10, :τ, :gammas, :μ))
chn_eq = sample(mod_eq, spl_eq, 5_000, discard_initial = 1_000);

# examine results for parameters
get_posterior_means_mu_sigma(mod_eq, chn_eq)
plotresults(mod_eq, chn_eq, D)
plotresults(mod_eq, chn_eq, x)




# examine results for the visited models
eq_samples = Int.(chn_eq[get_eq_ind_nms(chn_eq)].value.data); # samples of visited models
# frequency of unique models
countmap(vec(mapslices(x->join(Int.(x)), eq_samples, dims = 2)))
# posterior probability that equal_indices[i] == equal_indices[j], i.e., the post prob that two sds are equal.
compute_post_prob_eq(chn_eq)

# trace plots
gen_eq = generated_quantities(mod_eq, chn_eq);
τ = chn_eq[:τ].data
rhos = filter(startswith("g"), string.(chn_eq.name_map.parameters))
ρ2 = similar(chn_eq[rhos].value.data)
σ = similar(ρ2)
for i in eachindex(τ)
	σ[i, :] = gen_eq[i][1]
	ρ2[i, :] = gen_eq[i][2]
end

plots_sd  = [plot(σ[:, i],	title = "σ $i",	legend = false) for i in axes(σ, 2)]
plots_rho = [plot(ρ2[:, i],	title = "ρ $i", legend = false) for i in axes(ρ2, 2)]
plots_tau =  plot(τ, 		title = "τ", 	legend = false);

l = @layout [
	grid(2, k)
	a
]
plot(plots_sd..., plots_rho..., plots_tau, layout = l)

mean(view(σ, 1000:5000, :, :), dims = 1)

# @model function model(n::Vector{Float64}, b::Vector{Float64}, α::Vector{Float64}, ::Type{T} = Float64) where {T}

# 	k = 3 # only works for k = 3 in this example
# 	equal_indices = ones(Int, k)
# 	equal_indices[1] ~ Categorical([1])
# 	equal_indices[2] ~ Categorical([2/5, 3/5])
# 	if equal_indices[2] == 1
# 		equal_indices[3] ~ Categorical([.5, 0, .5])
# 	else
# 		equal_indices[3] ~ Categorical(1/3 .* ones(3))
# 	end

# 	τ ~ InverseGamma(1, 1)
# 	ρ ~ Dirichlet(α)
# 	ρ2 = Vector{T}(undef, k)
# 	for i in 1:k
# 		ρ2[i] = mean(ρ[equal_indices .== equal_indices[i]])
# 	end
# 	prec = ρ2 .* (τ * length(n))
# 	# adjust logposterior density
# 	inc_lpdf =
# 		# shortcut for a Jeffreys prior
# 		-logpdf(InverseGamma(1, 1), τ) +
# 		-log(τ) +
# 		# other terms for the likelihood
# 		sum(n .* log.(prec)) +
# 		-0.5 * sum(prec .* b)

# 	Turing.@addlogprob! inc_lpdf
# 	σ = length(n) * τ .* ρ2
# 	return (σ, ρ2)

# end

# sds = Float64[1, 1, 1]
# ns = 100 .* ones(Int, 3)

# k  = length(sds)
# ss = (sds .* ((ns .- 1) ./ ns)).^2
# n  = (ns .- 1) ./ 2
# b  = ns .* ss
# α  = ones(Float64, length(ss))

# mod = model(n, b, α)
# spl = Gibbs(PG(20, :equal_indices), HMC(0.005, 10, :τ, :ρ))
# chn = sample(mod, spl, 10_000)

# count frequency of models
eq_samples = chn[get_eq_ind_nms(chn)].value.data
countmap(vec(mapslices(x->join(Int.(x)), eq_samples, dims = 2)))
compute_post_prob_eq(chn)

# Trace plots
gen = generated_quantities(mod, chn)
rhos = filter(startswith("ρ"), string.(chn.name_map.parameters))
ρ2 = similar(chn[rhos].value.data)
σ = similar(ρ2)
for i in eachindex(gen)
	σ[i, :] = gen[i][1]
	ρ2[i, :] = gen[i][2]
end
τ = chn[:τ].data

plots_sd  = [plot(σ[:, i],	title = "σ $i",	legend = false) for i in axes(σ, 2)]
plots_rho = [plot(ρ2[:, i],	title = "ρ $i", legend = false) for i in axes(ρ2, 2)]
plots_tau =  plot(τ, 		title = "τ", 	legend = false);

l = @layout [
	grid(2, k)
	a
]
plot(plots_sd..., plots_rho..., plots_tau, layout = l)