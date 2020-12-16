import Distributions
get_eq_ind_nms(samples) = filter(x->startswith(string(x), "equal_indices"), samples.name_map.parameters)
function get_eq_samples(samples)
	eq_ind_nms = get_eq_ind_nms(samples)
	s = size(samples[eq_ind_nms])
	return reshape(samples[eq_ind_nms].value.data, :, s[2])
end


"""
	compute the proportion of samples where equal_indices[i] == equal_indices[j] ∀i, j
"""
function compute_post_prob_eq(samples)
	samps = get_eq_samples(samples)
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

"""
	compute how often each model is visited
"""
function compute_model_counts(chn)
	eq_samples = get_eq_samples(chn)
	return sort(countmap(vec(mapslices(x->join(reduce_model(Int.(x))), eq_samples, dims = 2))))
end

"""
	compute the posterior probability of each model
"""
function compute_model_probs(chn)
	count_models = compute_model_counts(chn)
	total_visited = sum(values(count_models))
	probs_models = Dict{String, Float64}()
	for (model, count) in count_models
		probs_models[model] = count / total_visited
	end
	return sort(probs_models)
end

"""
	compute the posterior probability of including 0, ..., k-1 equalities in the model
"""
function compute_incl_probs(chn)
	eq_samples = get_eq_samples(chn)
	k = size(eq_samples)[2]
	inclusions_per_model = vec(mapslices(x->k - length(Set(x)), eq_samples, dims = 2))
	count = countmap(inclusions_per_model)
	return sort(counts2probs(count))
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
function plotresults(model, chain, D::Distributions.MvNormal)
	μ, σ = get_posterior_means_mu_sigma(model, chain)
	plotresultshelper(μ, σ, D.μ, sqrt.(D.Σ.diag))
end

"""
	plot observed values vs posterior means
"""
function plotresults(model, chain, x::AbstractMatrix)
	μ, σ = get_posterior_means_mu_sigma(model, chain)
	plotresultshelper(μ, σ, vec(mean(x, dims = 1)), vec(sqrt.(var(x, dims = 1))))
end

function plotresultshelper(μ, σ, obsμ, obsσ)
	plot_μ = scatter(obsμ, μ, title = "μ", legend = false);
	Plots.abline!(plot_μ, 1, 0);
	plot_σ = scatter(obsσ, σ, title = "σ", legend = false);
	Plots.abline!(plot_σ, 1, 0);
	plot(plot_μ, plot_σ, layout = (2, 1))
end

function plottrace(mod, chn)

	gen = generated_quantities(mod, chn);
	τ = chn[:τ].data
	rhos = filter(startswith("ρ"), string.(chn.name_map.parameters))
	ρ2 = similar(chn[rhos].value.data)
	σ = similar(ρ2)
	for i in eachindex(τ)
		σ[i, :] = gen[i][1]
		ρ2[i, :] = gen[i][2]
	end

	plots_sd  = [plot(σ[:, i],	title = "σ $i",	legend = false) for i in axes(σ, 2)]
	plots_rho = [plot(ρ2[:, i],	title = "ρ $i", legend = false) for i in axes(ρ2, 2)]
	plots_tau =  plot(τ, 		title = "τ", 	legend = false);

	l = @layout [
		grid(2, k)
		a
	]
	return plot(plots_sd..., plots_rho..., plots_tau, layout = l)
end

function counts2probs(counts::Dict{T, Int}) where T
	total_visited = sum(values(counts))
	probs = Dict{T, Float64}()
	for (model, count) in counts
		probs[model] = count / total_visited
	end
	return probs
end

function count_equalities(sampled_models)

	k, n = size(sampled_models)
	result = Vector{Int}(undef, n)
	for i in axes(sampled_models, 2)
		result[i] = k - length(unique(view(sampled_models, :, i)))
	end
	return result
end

function plot_modelspace(D, probs_models_by_incl)

	k = length(D)
	y = expected_model_probabilities(D)
	incl_size = expected_inclusion_counts(k)
	cumulative_sum = 0
	xstart = 0
	plt = bar(probs_models_by_incl, legend=false, yaxis=:log);
	for i in 1:k
		yval = y[cumulative_sum + incl_size[i]]
		ycoords = [yval, yval]
		if incl_size[i] == 1
			xcoords = [xstart + 0.5, xstart + 0.5]
			scatter!(plt, xcoords, ycoords, m = 4);
		else
			xcoords = [xstart, xstart + incl_size[i]]
			plot!(plt, xcoords, ycoords, lw = 4);
		end
		cumulative_sum += incl_size[i]
		xstart += incl_size[i]
	end
	return plt
end

function plot_inclusionprobabilities(D, probs_equalities)
	k = length(D)
	plt = bar(sort(probs_equalities), legend=false, yaxis=:log);
	scatter!(plt, 0:k-1, expected_inclusion_probabilities(D), m = 4);
	return plt
end

function plot_expected_vs_empirical(D, probs_models_by_incl)
	x, y = expected_model_probabilities(D), collect(values(probs_models_by_incl))
	ablinecoords = [extrema([extrema(x)..., extrema(y)...])...]
	plt = plot(ablinecoords, ablinecoords, lw = 3, legend = false, yaxis=:log, xaxis=:log);
	plot!(plt, x, y, seriestype = :scatter);
end

function counts2probs(counts::Dict{T, Int}) where T
	total_visited = sum(values(counts))
	probs = Dict{T, Float64}()
	for (model, count) in counts
		probs[model] = count / total_visited
	end
	return probs
end

function count_equalities(sampled_models)

	k, n = size(sampled_models)
	result = Vector{Int}(undef, n)
	for i in axes(sampled_models, 2)
		result[i] = k - length(unique(view(sampled_models, :, i)))
	end
	return result
end