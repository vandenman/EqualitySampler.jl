using Plots
import Distributions, LinearAlgebra

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
	axistype = any(<=(1e-8), y) ? :none : :log

	plt = plot(ablinecoords, ablinecoords, lw = 3, legend = false, yaxis=axistype, xaxis=axistype);
	plot!(plt, x, y, seriestype = :scatter);
end

function visualize_eq_samples(equalityPrior, empirical_model_probs, empirical_inclusion_probs)
	p1 = plot_modelspace(equalityPrior, empirical_model_probs);
	p2 = plot_inclusionprobabilities(equalityPrior, empirical_inclusion_probs);
	p3 = plot_expected_vs_empirical(equalityPrior, empirical_model_probs);
	pjoint = plot(p1, p2, p3, layout = grid(3, 1), title = ["log model probabilities" "log inclusion probability" "expected vs empirical log model probability"],
				  size = (600, 1200))
end

function visualize_helper(prior_distribution, samples)
	println("equality matrix")
	display(LinearAlgebra.UnitLowerTriangular(compute_post_prob_eq(samples)))
	empirical_model_probs = compute_model_probs(samples)
	empirical_inclusion_probs = compute_incl_probs(samples)
	visualize_eq_samples(prior_distribution, empirical_model_probs, empirical_inclusion_probs)
end
