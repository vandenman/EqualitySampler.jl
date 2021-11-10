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
import Plots.PlotMeasures: mm

import Printf
round_2_decimals(x::Number) = Printf.@sprintf "%.2f" x
round_2_decimals(x) = x

EqualitySampler.logpdf_model_distinct(D::BetaBinomial, no_equalities::Integer) = logpdf(D, no_equalities) - log(binomial(promote(ntrials(D), no_equalities)...))
diff_lpdf(D, no_inequalities) = logpdf_model_distinct(D, no_inequalities - 1) - logpdf_model_distinct(D, no_inequalities)

updateSize(d::BetaBinomial, k) = BetaBinomial(k, d.α, d.β)
updateSize(::UniformMvUrnDistribution, k) = UniformMvUrnDistribution(k)
# updateSize(d::BetaBinomialMvUrnDistribution, k) = BetaBinomialMvUrnDistribution(k, d.α, d.β)
updateSize(d::BetaBinomialMvUrnDistribution, k) = BetaBinomialMvUrnDistribution(k, k, d.β)
updateSize(d::RandomProcessMvUrnDistribution, k) = RandomProcessMvUrnDistribution(k, d.rpm)

make_title(d::BetaBinomial) = "Beta-binomial α=$(round_2_decimals(d.α)) β=$(round_2_decimals(d.β))"
make_title(::UniformMvUrnDistribution) = "Uniform"
make_title(d::BetaBinomialMvUrnDistribution) = "Beta-binomial α=$(round_2_decimals(d.α)) β=$(round_2_decimals(d.β))"
make_title(d::RandomProcessMvUrnDistribution) = "Dirichlet Process α=$(round_2_decimals(d.rpm.α))"

function scottberger_figure_1(d::Distribution, k::Integer = big(30))

	no_equalities = 0:k-1
	lpdf = logpdf_model_distinct.(Ref(d), no_equalities)

	# We reverse the number of equalities, add one, and label it the number of inequalities
	no_equalities = reverse(no_equalities)

	# figure 1
	plt = plot(no_equalities, lpdf, legend = false);
	if d isa UniformMvUrnDistribution
		ylims!(plt, (-60, -10));
	end
	xticks!(plt, 0:5:k);
	xlims!(plt, (-1, k+1));
	scatter!(plt, no_equalities, lpdf);
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

function matrix_plot(dists, k::Integer, pos_included = 2:2:k; include_log = false, legend_subplot_idx = nothing, legend_pos = :topleft, kwargs...)

	figures = Matrix{Plots.Plot}(undef, length(dists), include_log ? 3 : 2)
	Plots.resetfontsizes()
	Plots.scalefontsizes(1.6)

	for (i, d) in enumerate(dists)
		@show i, d
		figures[i, 1] = scottberger_figure_1(d, k)
		figures[i, 2] = scottberger_figure_2(d, pos_included; kwargs...)
		if include_log
			figures[i, 3] = scottberger_figure_2(d, pos_included; give_log = true, kwargs...)
		end
	end

	# additional details
	x_middle = ceil(Int, length(dists) / 2)
	y_top = 1
	y_bottom = size(figures, 2)
	Plots.plot!(figures[x_middle, y_top], xlab = "Number of inequalities",    bottom_margin = 5mm);
	Plots.plot!(figures[x_middle, y_bottom], xlab = "Number of groups", bottom_margin = 5mm);
	Plots.plot!(figures[1, y_top], ylab = "Log probability", left_margin = 7mm);
	Plots.plot!(figures[1, 2], ylab = "Prior odds ratio (smaller / larger)", left_margin = 7mm);
	if include_log
		Plots.plot!(figures[1, 3], ylab = "Log prior odds ratio (smaller / larger)", left_margin = 7mm);
	end
	if !(legend_subplot_idx isa Bool && !legend_subplot_idx)

		if legend_subplot_idx isa Int && (1 < legend_subplot_idx < length(dists))
			legend_subplot_x_idx = legend_subplot_idx
		else
			legend_subplot_x_idx = x_middle
		end
		Plots.plot!(figures[legend_subplot_x_idx, y_bottom]; legend = legend_pos, foreground_color_legend = nothing, background_color_legend = nothing)
	end

	Plots.resetfontsizes()
	return figures
end

function make_jointplot(figures; width::Int = 500, kwargs...)
	ncols = size(figures, 1)
	nrows = size(figures, 2)
	return plot(figures...; layout = (nrows, ncols), size = (ncols * width, nrows * width), kwargs...)
end

function highlight_adjacent_points!(figure, d, x, label = "A", distance = 1.0; color = :grey, coordinates = nothing)

	x_rev = length(d) - 1 .- x
	A = logpdf_model_distinct.(Ref(d), x_rev)

	if isnothing(coordinates)
		slope = (A[2] - A[1]) / (x[2] - x[1])
		intercept = A[2] - x[2] * slope
		@assert A ≈ collect(x) .* slope .+ intercept

		# transform slope to account for aspectratio
		xlims = figure.subplots[1][:xaxis][:extrema]
		ylims = figure.subplots[1][:yaxis][:extrema]
		aspectratio = (ylims.emax - ylims.emin) / (xlims.emax - xlims.emin)
		# aspectratio

		μx, μA = mean(x), mean(A)
		perpendicular_slope = - 1 / (slope / aspectratio)
		perpendicular_intercept = μA - perpendicular_slope * μx
		@assert μA ≈ perpendicular_slope * μx + perpendicular_intercept

		function get_x2(d, b, m, x1, y1)
			discriminant = sqrt(-b^2+d^2+d^2 * m^2-2 * b * m * x1-m^2 * x1^2+2 * b * y1+2 * m * x1 * y1-y1^2)
			x2_1 = (-b * m + x1 + m * y1 - discriminant) / (1+m^2)
			x2_2 = (-b * m + x1 + m * y1 + discriminant) / (1+m^2)
			return x2_1, x2_2
		end

		x2_options = get_x2(distance, perpendicular_intercept, perpendicular_slope, μx, μA)
		y2_options = perpendicular_slope .* x2_options .+ perpendicular_intercept
		@assert all(x->x≈distance,  @. sqrt((x2_options - μx)^2 + (y2_options - μA)^2))

		# xx = 5:.1:15
		# plot(xx, slope .* xx .+ intercept, aspectratio = :equal)
		# plot!(xx, perpendicular_slope .* xx .+ perpendicular_intercept)
		# scatter!([μx], [μA], marker = (5, 1.0, :green))
		# scatter!([x2], [y2], marker = (5, 1.0, :blue))

		# plt = deepcopy(figure)
		# plot!(plt, xx, slope .* xx .+ intercept)
		# plot!(xx, perpendicular_slope .* xx .+ perpendicular_intercept)
		# scatter!([μx], [μA], marker = (5, 1.0, :green))
		# scatter!([x2], [y2], marker = (5, 1.0, :blue))

		idx = 2 - (x2_options[1] > μx)
		x2, y2 = x2_options[idx], y2_options[idx]
	else
		@assert length(coordinates) == 2
		x2, y2 = coordinates
	end

	plot!(figure, x, A, linecolor = color)
	scatter!(figure, x, A; marker = (5, 1.0, color))
	scatter!(figure, [x2], [y2]; marker = (0, 0, :white), series_annotations = [label])

end

function highlight_adjacent_points2!(figure, d, variables_added, label = "A", y_nudge = 0.0; bg_color = :grey)

	D = updateSize(d, length(d))
	x = fill(Float64(length(D)), length(variables_added))
	y = exp.(diff_lpdf.(Ref(D), variables_added))
	scatter!(figure, x, y; marker = (5, 1.0, bg_color), label = "")
	scatter!(figure, x, y .+ y_nudge; marker = (0, 0, :white), series_annotations = label, label = "")

end

k = BigInt(20)
variables_added = [1, 2, 5, 10]
pos_included = 2:2:30
# labels = [" 1 inequality added" " 2 inequalities added" " 5 inequalities added" "10 inequalities added"]
labels = ["p(#0) / p(#1)" "p(#1) / p(#2)" "p(#4) / p(#5)" "p(#9) / p(#10)"]

dists = (
	BetaBinomial(k, 1, 1),
	DirichletProcessMvUrnDistribution(k, 0.5),
	BetaBinomialMvUrnDistribution(k, Int(k), 1),
	UniformMvUrnDistribution(k)
)

figures = matrix_plot(dists, k, pos_included; label = labels, legend_subplot_idx = 3)
joint_2x4 = make_jointplot(figures, legendfont = font(12), titlefont = font(20));
# savefig(joint_2x4, joinpath("figures", "prior_comparison_plot_2x4_without_log.pdf"))

variables_added = [1, 2, 5, 10]
pos_included = 2:2:30
dists = (
	DirichletProcessMvUrnDistribution(k, 0.5),
	BetaBinomialMvUrnDistribution(k, Int(k), 1),
	UniformMvUrnDistribution(k)
)
figures = matrix_plot(dists, k, pos_included; label = labels, legend_subplot_idx = 3)
plot!(figures[3, 2], ylim = (0, 2));
plot!(figures[1, 1], ylim = (-60, 0), yticks = [-60, -40, -20, 0]);
joint_2x3 = make_jointplot(figures, legendfont = font(12), titlefont = font(20), tickfont = 12);
savefig(joint_2x3, joinpath("figures", "prior_comparison_plot_2x3_without_log_without_betabinomial.pdf"))

# add A and B to figures in the top row
highlight_adjacent_points!(figures[1, 1], dists[1], 0:1, "A");
highlight_adjacent_points!(figures[1, 1], dists[1], 9:10, "B"; coordinates = (9.8, -43));
highlight_adjacent_points!(figures[2, 1], dists[2], 0:1, "A");
highlight_adjacent_points!(figures[2, 1], dists[2], 9:10, "B"; coordinates = (9.25, -31.5));
# add A and B to figures in the bottom row
highlight_adjacent_points2!(figures[1, 2], dists[1], [1, 10], ["A", "B"], .03);
highlight_adjacent_points2!(figures[2, 2], dists[2], [1, 10], ["A", "B"], .8);

joint_2x3_with_points = make_jointplot(figures, legendfont = font(12), titlefont = font(20), guidefont = font(18), tickfont = font(12));
savefig(joint_2x3_with_points, joinpath("figures", "prior_comparison_plot_2x3_without_log_without_betabinomial_with_points.pdf"))

αs = (0.10, 0.25, 0.50, 1.00, 2.50, 5.00)
dists = Tuple(DirichletProcessMvUrnDistribution.(k, αs))
figures = matrix_plot(dists, k, pos_included; label = labels, legend_subplot_idx = 3, legend_pos = :bottomleft)
joint_2x6 = make_jointplot(figures, legendfont = font(12), titlefont = font(20));
savefig(joint_2x6, joinpath("figures", "prior_comparison_plot_2x6_without_log_dpp_only.pdf"))

pdf_model_distinct.(dists, 0)
round.(exp.(diff_lpdf.(dists, 1)); digits = 2)