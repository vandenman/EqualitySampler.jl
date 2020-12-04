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
	return countmap(vec(mapslices(x->join(reduce_model(Int.(x))), eq_samples, dims = 2)))
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
	return probs_models
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

"""
	reduce a model to a unique representation. For example, [2, 2, 2] -> [1, 1, 1]
"""
function reduce_model(x::Vector{T}) where T <: Integer

	y = copy(x)
	for i in eachindex(x)
		if !any(==(x[i]), x[1:i - 1])
			if x[i] > i
				idx = findall(==(x[i]), x[i:end]) .+ i .- 1 
				y[idx] .= i
			elseif x[i] < i
				y[i] = i
			end
		end
	end
	return y
end