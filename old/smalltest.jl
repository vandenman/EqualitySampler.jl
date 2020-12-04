using Turing, StatsBase, DynamicPPL, FillArrays, Plots

get_eq_ind_nms(samples) = filter(x->startswith(string(x), "equal_indices"), samples.name_map.parameters)

function trace_plots(chn)
	parameters = chn.name_map.parameters
	plots = [plot(chn[p].data, title = string(p), legend = false) for p in parameters]
	plot(plots..., layout = length(parameters))
end

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

@model function model(x, sample_equalities, ::Type{T} = Float64) where {T}

	# sampling equalities only works for k = 3 in this example
	k, n = size(x)

	τ ~ InverseGamma(15, 3)
	ρ ~ Dirichlet(ones(k))
	μ ~ filldist(Normal(0, 10), k)

	if sample_equalities

		# sample equalities among the sds
		equal_indices = ones(Int, k)
		equal_indices[1] ~ Categorical([1])
		equal_indices[2] ~ Categorical([2/5, 3/5])
		equal_indices[3] ~ Categorical(equal_indices[2] == 1 ? [.5, 0, .5] : 1/3 .* ones(3))

		# reassign rho to conform to the sampled equalities
		ρ2 = Vector{T}(undef, k)
		for i in 1:k
			# two sds are equal if equal_indices[i] == equal_indices[j]
			# if they're equal, ρ2[i] is the average of all ρ[j] ∀ j ∈ 1:k such that equal_indices[i] == equal_indices[j]
			ρ2[i] = mean(ρ[equal_indices .== equal_indices[i]])
		end
		# k * τ .* ρ2 gives the precisions of each group
		σ = 1 ./ sqrt.(k * τ .* ρ2)
	else # do not sample equalities
		# k * τ .* ρ  gives the precisions of each group
		σ = 1 ./ sqrt.(k * τ .* ρ)
	end

	# likelihood
	for i in 1:n
		x[:, i] ~ MvNormal(μ, σ)
	end

	if sample_equalities
		return (σ, ρ2)
	else
		return (σ,   )
	end
end

# simulate data
n = 1000
k = 3
sds = Float64[1, 1, 1]
D = MvNormal(sds)
x = rand(D, n)

# fit model without equalities as a rationality check
mod_no_eq = model(x, false)
spl_no_eq = Gibbs(HMC(0.004, 10, :τ, :ρ, :μ))
chn_no_eq = sample(mod_no_eq, spl_no_eq, 2_000);
# chn_no_eq = sample(mod_no_eq, NUTS(), 2_000);

trace_plots(chn_no_eq)

# examine results for μ & σ
summarystats_no_eq = summarystats(chn_no_eq)
μ_means_no_eq = summarystats_no_eq.nt.mean[startswith.(string.(summarystats_no_eq.nt.parameters), "μ")]
gen_no_eq = generated_quantities(mod_no_eq, chn_no_eq)
σ_means_no_eq = collect(mean(first(x)[j] for x in gen_no_eq) for j in 1:k)

# plot true values vs posterior means
plot_μ = scatter(1:k, μ_means_no_eq, title = "μ", legend = false);
Plots.abline!(plot_μ, 0, 0);
plot_σ = scatter(sds, σ_means_no_eq, title = "σ", legend = false);
Plots.abline!(plot_σ, 1, 0);
plot(plot_μ, plot_σ, layout = (2, 1))

# observed values vs posterior means
plot_μ = scatter(mean(x, dims = 2), μ_means_no_eq, title = "μ", legend = false);
Plots.abline!(plot_μ, 1, 0);
plot_σ = scatter(sqrt.(var(x, dims = 2)), σ_means_no_eq, title = "σ", legend = false);
Plots.abline!(plot_σ, 1, 0);
plot(plot_μ, plot_σ, layout = (2, 1))


# fit model with equalities
mod_eq = model(x, true)
spl_eq = Gibbs(PG(20, :equal_indices), HMC(0.02, 10, :τ, :ρ, :μ))
chn_eq = sample(mod_eq, spl_eq, 2_000)

# examine results for μ & σ
summarystats_eq = summarystats(chn_eq)
μ_means_eq = summarystats_eq.nt.mean[startswith.(string.(summarystats_eq.nt.parameters), "μ")]
gen_eq = generated_quantities(mod_eq, chn_eq)
σ_means_eq = collect(mean(first(x)[j] for x in gen_eq) for j in 1:k)

# plot true values vs posterior means
plot_μ = scatter(1:k, μ_means_eq, title = "μ",	legend = false);
Plots.abline!(plot_μ, 0, 0)
plot_σ = scatter(sds, σ_means_eq, title = "σ",	legend = false);
Plots.abline!(plot_σ, 1, 0)
plot(plot_μ, plot_σ, layout = (2, 1))

# observed values vs posterior means
plot_μ = scatter(mean(x, dims = 2), μ_means_eq, title = "μ", legend = false);
Plots.abline!(plot_μ, 1, 0);
plot_σ = scatter(sqrt.(var(x, dims = 2)), σ_means_eq, title = "σ", legend = false);
Plots.abline!(plot_σ, 1, 0);
plot(plot_μ, plot_σ, layout = (2, 1))

# examine results for the visited models
eq_samples = Int.(chn_eq[get_eq_ind_nms(chn_eq)].value.data); # samples of visited models
# frequency of unique models
countmap(vec(mapslices(x->join(Int.(x)), eq_samples, dims = 2)))
# posterior probability that equal_indices[i] == equal_indices[j], i.e., the post prob that two sds are equal.
compute_post_prob_eq(chn_eq)

# trace plots
τ = chn_eq[:τ].data;
rhos = filter(startswith("ρ"), string.(chn_eq.name_map.parameters))
ρ2 = similar(chn_eq[rhos].value.data);
σ = similar(ρ2);
for i in eachindex(τ)
	σ[i, :] = gen_eq[i][1]
	ρ2[i, :] = gen_eq[i][2]
end

idx = 1001:2000

plots_sd  = [plot(σ[idx, i],	title = "σ $i",	legend = false) for i in axes(σ, 2)];
plots_rho = [plot(ρ2[idx, i],	title = "ρ $i", legend = false) for i in axes(ρ2, 2)];
plots_tau =  plot(τ[idx], 		title = "τ", 	legend = false);

l = @layout [
	grid(2, k)
	a
];
plot(plots_sd..., plots_rho..., plots_tau, layout = l)

plot(plots_sd...)
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

using Turin, Optim
@model gibbs_example(x) = begin
	v1 ~ Normal(0,1)
	v2 ~ Categorical(5)
end
spl = Gibbs(HMC(0.2, 3, :v1), PG(20, :v2))
m = gibbs_example(1)
map_estimate = optimize(m, MAP())
chain = sample(m, spl, 10, init_theta = map_estimate.values.array)


chain = sample(m, spl, 10)

