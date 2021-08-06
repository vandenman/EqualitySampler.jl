#=

	This file replicates figures 1 and 2 of:

	Scott, J. G., & Berger, J. O. (2010). Bayes and empirical-Bayes multiplicity adjustment in the variable-selection problem. The Annals of Statistics, 2587-2619.

	in addition, it repeats the same simulation for the following prior distributions over partitions:

		UniformConditionalUrnDistribution
		BetaBinomialConditionalUrnDistribution
		DirichletProcessDistribution (Chinese restaurant)


	# TODO: double check exactly what Scott & Berger did!

=#

using EqualitySampler, Plots, Distributions
import Turing


EqualitySampler.logpdf_model_distinct(D::BetaBinomial, no_equalities::Integer) = logpdf(D, no_equalities) - log(binomial(promote(ntrials(D), no_equalities)...))
diff_lpdf(D, no_inequalities) = logpdf_model_distinct(D, no_inequalities - 1) - logpdf_model_distinct(D, no_inequalities)

updateSize(d::BetaBinomial, k) = BetaBinomial(k, d.α, d.β)
updateSize(::UniformMvUrnDistribution, k) = UniformMvUrnDistribution(k)
updateSize(d::BetaBinomialMvUrnDistribution, k) = BetaBinomialMvUrnDistribution(k, d.α, d.β)
updateSize(d::RandomProcessMvUrnDistribution, k) = RandomProcessMvUrnDistribution(k, d.rpm)

make_title(d::BetaBinomial) = "Beta-binomial α=$(d.α) β=$(d.β)"
make_title(::UniformMvUrnDistribution) = "Uniform"
make_title(d::BetaBinomialMvUrnDistribution) = "Beta-binomial α=$(d.α) β=$(d.β)"
make_title(d::RandomProcessMvUrnDistribution) = "Dirichlet Process α=$(d.rpm.α)"

function scottberger_figure_1(d::Distribution, k::Integer = big(30))

	included = 0:k
	lpdf = logpdf_model_distinct.(Ref(d), included)

	# figure 1
	plt = plot(included, lpdf, legend = false);
	if d isa UniformMvUrnDistribution
		ylims!(plt, (-60, -10));
	end
	xticks!(plt, 0:5:k);
	xlims!(plt, (-1, k+1));
	scatter!(plt, included, lpdf);
	title!(plt, make_title(d));

	return plt
end

function scottberger_figure_2(
		d::Distribution,
		pos_included = big(2):2:30;
		variables_added = big.((1, 2, 5, 10)),
		legend = false,
		give_log::Bool = false,
		label = missing
)

	result = Matrix{Float64}(undef, length(variables_added), length(pos_included))
	for (i, p) in enumerate(pos_included)
		D = updateSize(d, p)
		for j in eachindex(variables_added)
			result[j, i] = diff_lpdf(D, variables_added[j])
		end
	end

	if ismissing(label)
		label = permutedims(["$p included" for p in variables_added])
	end

	plt = plot(pos_included, give_log ? result' : exp.(result)'; label = label, legend = legend)

	return plt

end

function matrix_plot(dists, k::Integer, pos_included = 2:2:k; kwargs...)

	figures = Matrix{Plots.Plot}(undef, length(dists), 3)
	for (i, d) in enumerate(dists)
		@show i, d
		figures[i, 1] = scottberger_figure_1(d, k)
		figures[i, 2] = scottberger_figure_2(d, pos_included; kwargs...)
		figures[i, 3] = scottberger_figure_2(d, pos_included; give_log = true, kwargs...)
	end
	return figures
end

function make_figs(dists)
	figs = Matrix{Plots.Plot}(undef, 3, length(dists))
	for (i, d) in enumerate(dists)
		@show i, d

		# if d isa RandomProcessDistribution
		# 	included = BigInt(0):k
		# else
			included = 0:k
		# end
		lpdf = logpdf_model.(Ref(d), included)

		# @assert isapprox(sum(exp, lpdf), 1.0, atol = 1e-4) # <- should also be multiplied with number of models

		fig1 = plot(included, lpdf, legend = false);
		if d isa UniformMvUrnDistribution
			ylims!(fig1, (-60, -10));
		end
		xticks!(fig1, 0:5:k);
		xlims!(fig1, (-1, k+1));
		scatter!(fig1, included, lpdf);
		title!(fig1, make_title(d));

		result = Matrix{Float64}(undef, length(variables_added), length(pos_included))
		for (i, p) in enumerate(pos_included)
			D = updateSize(d, p)
			for j in eachindex(variables_added)
				result[j, i] = diff_lpdf(D, variables_added[j])
			end
		end

		fig2 = plot(pos_included, exp.(result)', label = permutedims(["$p included" for p in variables_added]),
					legend = isone(i) ? :topleft : :none);
		# title!(fig2, "Fig 2 of S & B");

		fig3 = plot(pos_included, result', label = permutedims(["$p included" for p in variables_added]),
		legend = isone(i) ? :topleft : :none);
		# title!(fig3, "Fig 2 of S & B (log scale)");
		if d isa UniformMvUrnDistribution
			plot!(fig2, ylims = (0, 2),  yticks = [0, 1, 2])
			plot!(fig3, ylims = (-1, 1), yticks = [-1, 0, 1])
		end

		# if i == length(dists)
		# 	plot!(twinx(fig1), tick = nothing, ylabel = "Figure 1 of S & B", ticks = nothing)
		# 	plot!(twinx(fig2), tick = nothing, ylabel = "Figure 2 of S & B", )
		# 	plot!(twinx(fig3), tick = nothing, ylabel = "Figure 3 of S & B (log scale)")
		# end

		figs[1, i] = fig1
		figs[2, i] = fig2
		figs[3, i] = fig3

	end
	return figs
end

function make_jointplot(figures; width::Int = 500)
	ncols = size(figures, 1)
	nrows = size(figures, 2)
	return plot(figures..., layout = (nrows, ncols), size = (ncols * width, nrows * width))
end

k = BigInt(30)
variables_added = [1, 2, 5, 10]
pos_included = 2:2:30

dists = (
	BetaBinomial(k, 1, 1),
	DirichletProcessMvUrnDistribution(k, 0.5),
	UniformMvUrnDistribution(k),
	BetaBinomialMvUrnDistribution(k, 1, 1)
	# RandomProcessMvUrnDistribution(k, Turing.RandomMeasures.DirichletProcess(1.887))
)

labels = reshape(["1ˢᵗ", "2ⁿᵈ", "3ʳᵈ", "4ᵗʰ"] .* " variable added", 1, 4)
figures = matrix_plot(dists, k; label = labels)
beep()

legend_on!(x)  = plot!(x; legend = :topleft, foreground_color_legend = nothing, background_color_legend = nothing)
# legend_off!(x) = plot!(x; legend = false)

legend_on!(figures[3, 3])
joint_3x4 = make_jointplot(figures)
legend_on!(figures[3, 2])
joint_2x4 = make_jointplot(figures[:, 1:2])
joint_2x3 = make_jointplot(figures[2:4, 1:2])

savefig(joint_3x4, joinpath("figures", "prior_comparison_plot_3x4_with_log.pdf"))
savefig(joint_2x4, joinpath("figures", "prior_comparison_plot_2x4_without_log.pdf"))
savefig(joint_2x3, joinpath("figures", "prior_comparison_plot_2x4_without_log_without_betabinomial.pdf"))
