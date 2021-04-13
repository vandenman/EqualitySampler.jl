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

k = 30#BigInt(50)
variables_added = [1, 2, 5, 10]
pos_included = 2:2:30

dists = [
	BetaBinomial(k, 1, 1),
	UniformMvUrnDistribution(k),
	BetaBinomialMvUrnDistribution(k, 1, 1),
	RandomProcessMvUrnDistribution(k, Turing.RandomMeasures.DirichletProcess(1.887))
]
updateSize(d::BetaBinomial, k) = BetaBinomial(k, d.α, d.β)
updateSize(::UniformMvUrnDistribution, k) = UniformMvUrnDistribution(k)
updateSize(d::BetaBinomialMvUrnDistribution, k) = BetaBinomialMvUrnDistribution(k, d.α, d.β)
updateSize(d::RandomProcessMvUrnDistribution, k) = RandomProcessMvUrnDistribution(k, d.rpm)

figs = Matrix{Plots.Plot}(undef, 3, length(dists))
for (i, d) in enumerate(dists)
	@show i, d

	if d isa RandomProcessDistribution
		included = BigInt(0):k
	else
		included = 0:k
	end
	lpdf = logpdf_model.(Ref(d), included)
	# @assert sum(exp, lpdf) ≈ 1

	fig1 = plot(included, lpdf, legend = false);
	if d isa UniformMvUrnDistribution
		ylims!(fig1, (-60, -10));
	end
	xticks!(fig1, 0:5:k);
	xlims!(fig1, (-1, k+1));
	scatter!(fig1, included, lpdf);
	title!(fig1, "$(typeof(d).name.name)");

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

	if i == length(dists)
		plot!(twinx(fig1), tick = nothing, ylabel = "Figure 1 of S & B", ticks = nothing)
		plot!(twinx(fig2), tick = nothing, ylabel = "Figure 2 of S & B", )
		plot!(twinx(fig3), tick = nothing, ylabel = "Figure 3 of S & B (log scale)")
	end

	figs[1, i] = fig1
	figs[2, i] = fig2
	figs[3, i] = fig3

end

w = 500
ncols = size(figs, 2)
jointPlot = plot(permutedims(figs)..., layout = (3, ncols), size = (3w, w * ncols))
# savefig(jointPlot, "figures")


# plot(figs[2, 2], size = (300, 300))

# d = dists[4]
# D = updateSize(d, 5)

# pdf_model.(Ref(D), 0:4)
# pdf_model.(Ref(D), 0:4)

# pdf_incl.(Ref(D), 0:4)

# cc = EqualitySampler.count_distinct_models_with_incl.(5, 0:4)
# pdf_incl.(Ref(D), 0:4) ./ cc

