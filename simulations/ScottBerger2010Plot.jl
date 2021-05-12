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


EqualitySampler.logpdf_model(D::BetaBinomial, no_equalities::Integer) = logpdf(D, no_equalities) - log(binomial(promote(ntrials(D), no_equalities)...))
diff_lpdf(D, no_inequalities) = logpdf_model(D, no_inequalities - 1) - logpdf_model(D, no_inequalities)

updateSize(d::BetaBinomial, k) = BetaBinomial(k, d.α, d.β)
updateSize(::UniformMvUrnDistribution, k) = UniformMvUrnDistribution(k)
updateSize(d::BetaBinomialMvUrnDistribution, k) = BetaBinomialMvUrnDistribution(k, d.α, d.β)
updateSize(d::RandomProcessMvUrnDistribution, k) = RandomProcessMvUrnDistribution(k, d.rpm)

make_title(d::BetaBinomial) = "BetaBinomial α=$(d.α) β=$(d.β)"
make_title(::UniformMvUrnDistribution) = "Uniform"
make_title(d::BetaBinomialMvUrnDistribution) = "BetaBinomial α=$(d.α) β=$(d.β)"
make_title(d::RandomProcessMvUrnDistribution) = "Dirichlet Process α=$(d.rpm.α)"

function scottberger_figure_1(d::Distribution, k::Integer = big(30))

	included = 0:k
	lpdf = logpdf_model.(Ref(d), included)

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
		k::Integer = big(30),
		variables_added = big.((1, 2, 5, 10)),
		pos_included = big(2):2:30;
		legend = false
)

	result = Matrix{Float64}(undef, length(variables_added), length(pos_included))
	for (i, p) in enumerate(pos_included)
		D = updateSize(d, p)
		for j in eachindex(variables_added)
			result[j, i] = diff_lpdf(D, variables_added[j])
		end
	end

	plt = plot(pos_included, exp.(result)', label = permutedims(["$p included" for p in variables_added]);
				legend = legend);

	return plt

end

function matrix_plot(dists)

	figures = Matrix{Plots.Plot}(undef, length(dists), 2)
	for (i, d) in enumerate(dists)
		@show i, d
		figures[i, 1] = scottberger_figure_1(d)
		figures[i, 2] = scottberger_figure_2(d)
	end
	return figures
end

k = 30#BigInt(50)
variables_added = [1, 2, 5, 10]
pos_included = 2:2:30

dists = [
	BetaBinomial(k, 1, 1),
	UniformMvUrnDistribution(k),
	BetaBinomialMvUrnDistribution(k, 1, 1),
	RandomProcessMvUrnDistribution(k, Turing.RandomMeasures.DirichletProcess(1.887))
]

figures = matrix_plot(dists)

w = 500
ncols = size(figures, 1)
jointPlot = plot(figures..., layout = (2, ncols), size = (2w, w * (ncols - 2)))
savefig(jointPlot, joinpath("figures", "prior_comparison_plot_2x4_without_log.pdf"))


# TODO: wrap this in a function
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

w = 500
ncols = size(figs, 2)
jointPlot = plot(permutedims(figs)..., layout = (3, ncols), size = (3w, w * ncols))
savefig(jointPlot, joinpath("figures", "prior_comparison_plot_3x4_with_log.pdf"))

jointPlot = plot(permutedims(view(figs, 1:2, :))..., layout = (2, ncols), size = (2w, w * (ncols - 2)))
savefig(jointPlot, joinpath("figures", "prior_comparison_plot_2x4_without_log.pdf"))

figs2 = figs[1:2, 3:4]
plot!(figs2[2, 1], legend = :topleft)
jointPlot = plot(permutedims(figs2)..., layout = (2, 2), size = (2w, 2w))
savefig(jointPlot, joinpath("figures", "prior_comparison_plot_2x2_without_log.pdf"))



# diagnosing floating point issues with DPP...

# d = dists[4]
# included = 0:k
# lpdf = logpdf_model.(Ref(d), included)

# plot(included, lpdf, legend = false)
# show(stdout, "text/plain", hcat(included, lpdf))

# logpdf_model(d, 22)
# logpdf_model(d, 23)
# logpdf_model(d, 24)

# logpdf_incl(d, 22)
# log_count_distinct_models_with_incl(length(d), 22)

# @code_warntype logunsignedstirlings1(30, 30-22)

# unsignedstirlings1(big(30), 22)
# log(unsignedstirlings1(big(30), 22))

# logunsignedstirlings1(30, 30-22)

# terms = map(r->EqualitySampler.stirlings1ExplLogTerm(n, k, r), 0:n-k)
# log(sum(exp, big.(terms)))

# nvals = 30
# kvals = 30
# for n in 1:nvals, k in 1:kvals
# 	# @show n, k
# 	result = isapprox(logunsignedstirlings1(n, k), log(unsignedstirlings1(big(n), k)), atol = 1e-4)
# 	@show result, n, k
# end
# unsignedstirlings1(big(22), 2)

# (n, k) = (21, 3)
# logunsignedstirlings1(n, k) - log(unsignedstirlings1(big(n), k))

# terms = map(r->EqualitySampler.stirlings1ExplLogTerm(n, k, r), 0:n-k)
# log(sum(exp, big.(terms)))




# plot(figs[2, 2], size = (300, 300))

# d = dists[4]
# D = updateSize(d, 5)

# pdf_model.(Ref(D), 0:4)
# pdf_model.(Ref(D), 0:4)

# pdf_incl.(Ref(D), 0:4)

# cc = EqualitySampler.count_distinct_models_with_incl.(5, 0:4)
# pdf_incl.(Ref(D), 0:4) ./ cc

