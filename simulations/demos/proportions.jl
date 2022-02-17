using EqualitySampler, Turing, Plots, FillArrays, Plots.PlotMeasures, Colors
import	DataFrames		as DF,
		StatsBase 		as SB,
		LinearAlgebra	as LA,
		NamedArrays		as NA,
		CSV
import DynamicPPL: @submodel
import JLD2, KernelDensity, Printf, ColorSchemes, Colors
round_2_decimals(x::Number) = Printf.@sprintf "%.2f" x
round_2_decimals(x) = x

include("../anovaFunctions.jl")

journal_data = DF.DataFrame(CSV.File(joinpath("simulations", "demos", "data", "journal_data.csv")))

scatter(journal_data[!, :journal], journal_data[!, :errors], ylims = (0, 1), ylab = "Proportion of statistical reporting errors", label = nothing)

@assert journal_data[!, :errors] ≈ journal_data[!, :perc_articles_with_errors] ./ 100

make_title(d::BetaBinomial) = "Beta-binomial α=$(round_2_decimals(d.α)) β=$(round_2_decimals(d.β))"
make_title(::UniformMvUrnDistribution) = "Uniform"
make_title(d::BetaBinomialMvUrnDistribution) = "Beta-binomial α=$(round_2_decimals(d.α)) β=$(round_2_decimals(d.β))"
make_title(d::RandomProcessMvUrnDistribution) = "Dirichlet Process α=$(round_2_decimals(d.rpm.α))"
make_title(::Nothing) = "Full model"

function get_p_constrained(model, samps)

	default_result = generated_quantities(model, MCMCChains.get_sections(samps, :parameters))
	clean_result = Matrix{Float64}(undef, length(default_result[1][1]), size(default_result, 1))
	for i in eachindex(default_result)
		clean_result[:, i] .= default_result[i][1]
	end
	return vec(mean(clean_result, dims = 2)), clean_result
end

@model function proportion_model_full(no_errors, total_counts, partition = nothing, ::Type{T} = Float64) where T

	p_raw ~ filldist(Beta(1.0, 1.0), length(no_errors))
	p_constrained = isnothing(partition) ? p_raw : average_equality_constraints(p_raw, partition)
	no_errors ~ Distributions.Product(Binomial.(total_counts, p_constrained))
	return (p_constrained, )
end

@model function proportion_model_equality_selector(no_errors, total_counts, partition_prior)
	partition ~ partition_prior
	DynamicPPL.@submodel prefix="inner" p = proportion_model_full(no_errors, total_counts, partition)
	return p
end

function equality_prob_table(journal_data, samples)
	maxsize = maximum(length, journal_data[!, :journal])
	rawnms = ["$(rpad(journal, maxsize)) ($(round(prob, digits=3)))" for (journal, prob) in eachrow(journal_data[!, [:journal, :errors]])]
	nms = OrderedDict(rawnms .=> axes(journal_data, 1))
	return NA.NamedArray(
		collect(LA.UnitLowerTriangular(compute_post_prob_eq(samples))),
		(nms, nms), ("Rows", "Cols")
	)
end

function plot_observed_against_estimated(observed, estimated, label; markersize = 1, legend = :topleft, color = permutedims(distinguishable_colors(8, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)), xlab = "Observed proportion", ylab = "Posterior mean", kwargs...)
	plt = plot(;kwargs...);
	Plots.abline!(plt, 1, 0, legend = false, color = "black", label = nothing);
	scatter!(plt, observed', estimated', legend = legend, label = label, ylim = (0, 1), xlim = (0, 1), xlab = xlab, ylab = ylab, color = color, markersize = markersize)
	return plt
end

total_counts = journal_data[!, :n]
no_errors = round.(Int, journal_data[!, :x])

priors = (
	full = nothing,
	DPP1 = DirichletProcessMvUrnDistribution(length(no_errors), 0.5),
	DPP2 = DirichletProcessMvUrnDistribution(length(no_errors), 1.0),
	DPP3 = DirichletProcessMvUrnDistribution(length(no_errors), 2.0),
	Betabinomial1 = BetaBinomialMvUrnDistribution(length(no_errors), 1, 1),
	Betabinomial2 = BetaBinomialMvUrnDistribution(length(no_errors), 1, length(no_errors)),
	Betabinomial3 = BetaBinomialMvUrnDistribution(length(no_errors), 1, binomial(length(no_errors), 2)),
	Uniform = UniformMvUrnDistribution(length(no_errors))
)

fits_file = joinpath(pwd(), "simulations", "demos", "saved_objects", "proportions_fits.jld2")
if !isfile(fits_file)
	fits = map(priors) do prior

		if isnothing(prior)
			model = proportion_model_full(no_errors, total_counts)
			spl = NUTS()
		else
			model = proportion_model_equality_selector(no_errors, total_counts, prior)
			spl = Gibbs(
				HMC(0.05, 10, Symbol("inner.p_raw")),
				GibbsConditional(:partition, EqualitySampler.PartitionSampler(length(no_errors), get_logπ(model)))
			)
		end

		all_samples = sample(model, spl, 100_000);
		posterior_means, posterior_samples = get_p_constrained(model, all_samples)

		return (posterior_means=posterior_means, posterior_samples=posterior_samples, all_samples=all_samples, model=model)
	end
	JLD2.save_object(fits_file, fits)
else
	fits = JLD2.load_object(fits_file)
end

trace_plots = map(zip(priors, fits)) do (prior, fit)
	plot(fit[:posterior_samples]', xlab = "iteration", ylab = "proportions", legend = false, ylim = (0, 1), title = make_title(prior))
end

l = (isqrt(length(priors)), ceil(Int, length(priors) / isqrt(length(priors))))
joint_trace_plots = plot(trace_plots..., layout = l, size = 400 .* reverse(l))

cols = permutedims(distinguishable_colors(8, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)[1:8])
retrieval_plots = map(zip(priors, fits)) do (prior, fit)
	plot_observed_against_estimated(journal_data[!, :errors], fit[:posterior_means], permutedims(journal_data[!, :journal]); title = make_title(prior),
		legend = isnothing(prior) ? :topleft : nothing, foreground_color_legend = nothing, background_color_legend = nothing,
		markersize = 5, color = cols
	)
end
joint_retrieval_plots = plot(retrieval_plots..., layout = (1, length(priors)), size = (400length(priors), 400))
joint_retrieval_plots_2x2 = plot(retrieval_plots..., layout = (2, 2), size = (800, 800))
joint_retrieval_plots_full_bb = plot(retrieval_plots[[1, 3]]..., layout = (1, 2), size = (400*2, 400), bottom_margin = 5mm, left_margin = 5mm)

savefig(joint_trace_plots, "figures/demo_proportions_trace_plots.pdf")
savefig(joint_retrieval_plots, "figures/demo_proportions_retrieval_plots.pdf")
savefig(joint_retrieval_plots_2x2, "figures/demo_proportions_retrieval_plots_2x2.pdf")
savefig(joint_retrieval_plots_full_bb, "figures/demo_proportions_retrieval_plots_full_bb.pdf")

# count visited models
visited = map(zip(priors, fits)) do (prior, fit)

	isnothing(prior) && return nothing

	all_samples = fit[:all_samples]
	mp = sort(compute_model_probs(all_samples),  byvalue=true, rev=true)
	mc = sort(compute_model_counts(all_samples), byvalue=true, rev=true)

	return (
		count(!iszero, values(mp)),
		count(>(0), values(mc)), # number of models visited
		equality_prob_table(journal_data, all_samples)
	)

end

visited[4][1]

density_data = map(fits) do fit

	npoints = 2^12
	no_groups = size(fit.posterior_samples, 1)

	x = Matrix{Float64}(undef, npoints, no_groups)
	y = Matrix{Float64}(undef, npoints, no_groups)

	for (i, row) in enumerate(eachrow(fit.posterior_samples))
		k = KernelDensity.kde(row; npoints = npoints, boundary = (0, 1))
		x[:, i] .= k.x
		y[:, i] .= k.density
	end
	return (x = x, y = y)
end

nms = NamedTuple{keys(fits)}(keys(fits))

density_plots = map(nms) do (key)

	x, y = density_data[key]
	plt = plot(x, y, legend = false, title = make_title(priors[key]), xlim = (.25, .65))

end

legend_plot = plot(zeros(1, 8); showaxis = false, grid = false, axis = nothing,
	foreground_color_legend = nothing, background_color_legend = nothing,
	label = permutedims(journal_data[!, :journal]));
joined_density_plots = plot(density_plots..., legend_plot, size = (4, 2) .* 400, layout = @layout [grid(2,4) a{0.1w}]);
savefig(joined_density_plots, "figures/demo_proportions_joined_density_plots.pdf")

densityFull = deepcopy(density_data.full)
densityBB   = deepcopy(density_data.Betabinomial2)

# do not plot very small density values
densityFull.y .= ifelse.(densityFull.y .<= 0.1, NaN, densityFull.y)
densityBB.y   .= ifelse.(densityBB.y   .<= 0.1, NaN, densityBB.y)

my_theme = PlotThemes.PlotTheme(linewidth = 1.5)
PlotThemes.add_theme(:test, my_theme)
# Plots.showtheme(:test)
Plots.theme(:test)


axis_line_x = 0:0.01:1
axis_line_y = fill(0.0, length(axis_line_x))
xlim = (.25, .71)
ylim = (0, 70)
cols = permutedims(1:8)
plt1 = plot(densityFull.x, densityFull.y; xlim = xlim, ylim = ylim, title = make_title(priors.full), color = cols, label = permutedims(journal_data[!, :journal]), foreground_color_legend = nothing, background_color_legend = nothing);
plt2 = plot(densityBB.x, densityBB.y;     xlim = xlim, ylim = ylim, title = "Model averaged",        color = cols, legend = false, xlab = "Error probability");
# plot!(plt1, axis_line_x, axis_line_y, color = "black", linewidth = 1.0);
# plot!(plt2, axis_line_x, axis_line_y, color = "black", linewidth = 1.0);
left_panel = plot(plt1, plt2, layout = (2, 1));

plt_ylabel = plot([0 0]; ylab = "Density", showaxis = false, grid = false, axis = nothing, legend = false, left_margin = -6mm, right_margin = 6mm, ymirror=true)
plt_legend = plot(zeros(1, 8); showaxis = false, grid = false, axis = nothing, foreground_color_legend = nothing, background_color_legend = nothing, label = permutedims(journal_data[!, :journal]), legendtitle = "Journal",
	left_margin = 0mm);

left_panel = plot(plt_ylabel, plot(plt1, legend = false), plt2, plt_legend,
	bottom_margin = 3mm,
	layout = @layout [a{0.00001w} grid(2, 1) a{0.1w}]
);

left_panel = plot(plt_ylabel, plot(plt1, legend = (.95, .95), legendtitle = "Journal"), plt2,
	bottom_margin = 3mm,
	layout = @layout [a{0.00001w} grid(2, 1)]
);


eq_table = equality_prob_table(journal_data, fits.Betabinomial2.all_samples)
for i in 1:8, j in i:8
	eq_table[i, j] = NaN
end

# prior_eq_probs = mapreduce(+, eachcol(generate_distinct_models(8)); init = 0.0) do model
# 	if model[1] == model[2]
# 		return EqualitySampler.pdf_model_distinct(priors.Betabinomial2, model)
# 	else
# 		return 0.0
# 	end
# end

x_nms = journal_data[!, :journal]# names(eq_table)[1]
color_gradient = cgrad(cgrad(ColorSchemes.magma)[0.15:0.01:1.0])
annotations = []
for i in 1:7, j in i+1:8
	z = eq_table[9-i, 9-j]
	col = color_gradient[1 - z]
	push!(annotations,
		(
			8 - j + 0.5,
			i - 0.5,
			Plots.text(
				round_2_decimals(z),
				8, col, :center
			)
		)
	)
end

right_panel = heatmap(x_nms, reverse(x_nms), Matrix(eq_table)[8:-1:1, :],
	aspect_ratio = 1, showaxis = false, grid = false, color = color_gradient,
	clims = (0, 1),
	title = "Posterior probability of pairwise equality",
	#=colorbar_ticks = 0:.2:1, <- only works with pyplot =#
	annotate = annotations,
	xmirror = false);


joined_plot = plot(left_panel, right_panel, layout = (1, 2), size = (2, 1) .* 500);
savefig(joined_plot, "figures/demo_proportions_2panel_plot.pdf")

