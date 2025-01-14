using EqualitySampler
import AlgebraOfGraphics as AOG
import Distributions
import CairoMakie as CM
import	DataFrames		as DF,
		StatsBase 		as SB,
		LinearAlgebra	as LA,
		NamedArrays		as NA,
		CSV
import Printf, ColorSchemes, Colors, OrderedCollections
round_2_decimals(x::Number) = Printf.@sprintf "%.2f" x
round_2_decimals(x) = x

function equality_prob_table(journal_data, eq_prob_mat)
	maxsize = maximum(length, journal_data.journal)
	rawnms = ["$(rpad(journal, maxsize)) ($(round(prob, digits=3)))" for (journal, prob) in eachrow(journal_data[!, [:journal, :errors]])]
	nms = OrderedCollections.OrderedDict(rawnms .=> axes(journal_data, 1))
	return NA.NamedArray(
		collect(LA.UnitLowerTriangular(eq_prob_mat)),
		(nms, nms), ("Rows", "Cols")
	)
end
#endregion

# working directory is assumed to be the root of the GitHub repository
journal_data = DF.DataFrame(CSV.File(joinpath("simulations", "demos", "data", "journal_data.csv")))

# bar graph of the errors by journal
AOG.data(journal_data) *
	AOG.visual(CM.BarPlot) *
	AOG.mapping(:journal, :errors) |> AOG.draw()

# @assert journal_data[!, :errors] ≈ journal_data[!, :perc_articles_with_errors] ./ 100

# no. possible errors
total_counts = journal_data.n

# no. errors
no_errors = round.(Int, journal_data.x)

# no. MCMC iterations
no_iter = 200_000

# no. groups, i.e., K
no_journals = length(journal_data.journal)

partition_prior = BetaBinomialPartitionDistribution(no_journals, 1, no_journals)

# with EqualitySampler.EnumerateThenSample first enumerates the model space
# and then resamples to obtain model averaged parameter distributions.
prop_samples_eq   = proportion_test(total_counts, no_errors, EqualitySampler.EnumerateThenSample(iter = no_iter), partition_prior)

# compute the posterior probability of equality
eq_prop_mat = compute_post_prob_eq(prop_samples_eq)

# the posterior probability that two journals are equal
NA.NamedArray(
  LA.UnitLowerTriangular(round.(eq_prop_mat; digits = 2)),
  (journal_data.journal, journal_data.journal)
)

# a prettier table with names along the axes
eq_prob_table = equality_prob_table(journal_data, eq_prop_mat)
eq_prob_table
# 8×8 Named Matrix{Float64}
#  Rows ╲ Cols │ JAP  (0.336)  PS   (0.397)  JCCP (0.489)  PLOS (0.497)  FP   (0.508)  DP   (0.509)  JEPG (0.548)  JPSP (0.576)
# ─────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────
# JAP  (0.336) │          1.0           0.0           0.0           0.0           0.0           0.0           0.0           0.0
# PS   (0.397) │     0.097201           1.0           0.0           0.0           0.0           0.0           0.0           0.0
# JCCP (0.489) │  7.38333e-20    6.66749e-8           1.0           0.0           0.0           0.0           0.0           0.0
# PLOS (0.497) │  6.16726e-23     1.4969e-9      0.904766           1.0           0.0           0.0           0.0           0.0
# FP   (0.508) │  4.61693e-12    3.15509e-6      0.849854      0.860222           1.0           0.0           0.0           0.0
# DP   (0.509) │  3.93733e-27   2.74948e-11      0.852721      0.876882      0.870441           1.0           0.0           0.0
# JEPG (0.548) │  2.03908e-20   2.12541e-11     0.0692283     0.0749236      0.102925     0.0939783           1.0           0.0
# JPSP (0.576) │   2.8586e-57   7.04563e-32     2.886e-10     6.1213e-9    0.00611677    1.84968e-6      0.805901           1.0

# plot the posterior samples of the error probabilities
samples_df = DF.DataFrame(
    sample    = vcat(vec(prop_samples_full.parameter_samples.θ_p_samples), vec(prop_samples_eq.parameter_samples.θ_p_samples)),
    parameter = repeat(journal_data.journal, outer = 2no_iter),
    method    = repeat(["Full model", "Model averaged"], inner = no_iter * length(total_counts))
)

cols = Colors.parse.(Colors.Colorant, [
    "#009afa", "#e36f47", "#3da44e", "#c371d2",
    "#ac8e18", "#00aaae", "#ed5e93", "#c68225"
])
w = 650
f = CM.Figure(size = (2w, w), fontsize = 20)
left_panel  = f[1, 1] = CM.GridLayout()
right_panel = f[1, 2] = CM.GridLayout()

left_panel_aog = AOG.data(samples_df) *
    AOG.mapping(:sample => "Error probability", color = :parameter, layout = :method) *
    AOG.density()
# AOG.draw(left_panel_aog, axis = (ylabel = "Density", ))

legend_info = AOG.draw!(left_panel, left_panel_aog,
    axis = (ylabel = "Density", rightspinevisible = false, topspinevisible = false, titlefont = :regular,
    limits = ((.25, .7), nothing)),
    AOG.scales(
        Layout = (; palette = [(1, 1), (2, 1)]),
        Color = (;  palette = cols, categories = journal_data.journal)
    ))

AOG.legend!(f[1, 1], legend_info; tellwidth = false, tellheight = false,
    framevisible = false, backgroundcolor = :transparent,
    halign = :right, valign = :top, titlevisible = false, titlesize = 0.0f0)

centers_x = (axes(eq_prob_table, 1))
centers_y = reverse(axes(eq_prob_table, 2))
data      = Matrix(eq_prob_table)
for i in axes(data, 1), j in i:size(data, 2)
    data[i, j] = NaN
end
data = permutedims(data)

color_gradient = CM.cgrad(CM.cgrad(ColorSchemes.magma)[0.15:0.01:1.0])
ax = CM.Axis(right_panel[1, 1],
    title = "Posterior probability of pairwise equality", titlefont = :regular,
    xticks = (axes(eq_prob_table, 1), journal_data.journal),
    yticks = (axes(eq_prob_table, 1), reverse(journal_data.journal)),
    xgridvisible = false, ygridvisible = false)
    CM.hidespines!(ax)
hm = CM.heatmap!(ax, centers_x, centers_y, data, colormap = color_gradient, colorrange = (0, 1))
for i in axes(data, 2), j in i+1:size(data, 1)
    CM.text!(ax, centers_x[i], centers_y[j], text = round_2_decimals(data[i, j]), align = (:center, :center),
        color = color_gradient[1 - data[i, j]])
end
CM.Colorbar(right_panel[1, 2], limits = (0, 1), colormap = color_gradient, ticks = 0:.1:1)

f

CM.save("simulations/revision_figures/proportions_side_by_side.pdf", f)
