using Plots, Plots.PlotMeasures, DataFrames, Chain, Colors, ColorSchemes, Printf
using Statistics

include("priors_plot_colors_shapes_labels.jl")
include("simulation_helpers.jl")

results_joined_path = joinpath("simulations", "big_simulation_runs_joined", "big_simulation_runs_joined.jld2")
if !isfile(results_joined_path)
	results_dir = joinpath("simulations", "big_simulation_runs")
	results_df = read_results(results_dir)
	JLD2.jldsave(results_joined_path; results_df = results_df)
else
	tmp = JLD2.jldopen(results_joined_path)
	results_df = tmp["results_df"]
end

priors_to_remove = Set((:BetaBinomialk1, #=:DirichletProcessGP,=# :DirichletProcess2_0))
obs_per_group_to_keep = Set((50, 100, 250, 500))
reduced_results_df_ungrouped = @chain results_df begin
	subset(:obs_per_group => (x -> x .<= 500))
	filter(:prior => x -> x ∉ priors_to_remove, _)
	# filter(:obs_per_group => x-> x in obs_per_group_to_keep, _)
	groupby(Cols(:obs_per_group, :prior, :groups, :hypothesis))
	combine([:any_incorrect, :prop_incorrect, :α_error_prop, :β_error_prop] .=> mean, :α_error_prop => (x->mean(>(0.0), x)) => :any_α_error_prop)
	sort([order(:hypothesis, by=x->parse(Int, string(x)[2:end]), rev=true), order(:obs_per_group)])
end
reduced_results_df = groupby(reduced_results_df_ungrouped, Cols(:hypothesis, :groups))

reduced_results_averaged_df = @chain reduced_results_df_ungrouped begin
	groupby(Cols(:obs_per_group, :prior, :groups))
	combine([:any_α_error_prop, :β_error_prop_mean] .=> mean)
	groupby(:groups)
end


# tempdf = @chain results_df begin
# 	filter(:hypothesis => x-> x in (:p00, :p25, :p50), _)
# 	filter(:groups => x -> x == 5, _)
# 	# filter(:prior => x -> x === :BetaBinomial1binomk2, _)
# 	filter(:prior => x -> x === :Westfall, _)
# 	transform(:α_error_prop => (x->x .> 0.0) => :any_α_error_prop)
# 	sort([order(:hypothesis, by=x->parse(Int, string(x)[2:end]), rev=true), order(:obs_per_group)])
# 	# groupby(Cols(:prior, :hypothesis, :groups, :obs_per_group))
# 	# combine(:any_α_error_prop => mean)
# end
# tempdf[!, [1, 2, 4, 6, 8, 9, 10, 12, 18]]
# tempdf[4, :true_model]
# tempdf[4, :post_probs]
# show(stdout, "text/plain", tempdf[!, [1, 2, 4, 6, 8, 10, 12, 18]])
# prop_incorrect_αβ(results_df[6, :post_probs], results_df[6, :true_model], results_df[6, :prior] === :Westfall || results_df[6, :prior] === :Westfall_uncorrected)


reduced_results_df2 = @chain reduced_results_df_ungrouped begin
	filter(:obs_per_group => x-> x in obs_per_group_to_keep, _)
	groupby(Cols(:hypothesis, :groups))
end

function make_figure(df, y_symbol;
	# xticks = [50, 100, 250, 500, 750, 1000],
	# xlim = (0, 1050),
	xticks = [50, 100, 250, 500],
	xlim = (0, 550),
	kwargs...)
	# colors1 = get_colors(df[!, :prior], .8)
	colors2 = get_colors(df[!, :prior], .5)
	shapes = get_shapes(df[!, :prior])
	offset_size = 0.1
	offset_lookup = Dict(
		((:BetaBinomial11, :BetaBinomialk1, :BetaBinomial1k, :BetaBinomial1binomk2) .=> -offset_size)...,
		((:DirichletProcess0_5, :DirichletProcess1_0, :DirichletProcess2_0, :DirichletProcessGP) .=> offset_size)...,
		((:uniform, :Westfall, :Westfall_uncorrected) .=> 0.0)...
	)
	offset = [offset_lookup[prior] for prior in df[!, :prior]]
	Plots.plot(
		df[!, :obs_per_group] .+ offset,
		df[!, y_symbol],
		group = df[!, :prior],

		linecolor			= colors2,
		markercolor			= colors2,
		markershape			= shapes,
		# markerstrokealpha	= 0.0,
		markerstrokewidth	= 0,
		# markerstrokecolor	= colors1,
		labels				= get_labels(df[!, :prior]),
		markersize			= 7,
		markeralpha			= 0.75,

		# ylim   = (0, 1.05),
		# yticks = 0:.2:1,
		xticks = xticks,
		xlim   = xlim,
		# xlim   = (1, 11),

		background_color_legend = nothing,
		foreground_color_legend = nothing
		;
		kwargs...
	)
end

# ensure :p100 comes last
# keys_ordered = keys(reduced_results_df)[[1, 3, 4, 5, 2]]
# keys_ordered = keys(reduced_results_df)[[1, 3, 4, 5, 2, 6, 8, 9, 10, 7]]
keys_ordered = sort(keys(reduced_results_df), by = x-> 1000*x[2] + parse(Int, string(x[1])[2:end]))
keys_α = filter(x->x.hypothesis != :p100, keys_ordered)
keys_β = filter(x->x.hypothesis != :p00,  keys_ordered)
key_to_title(x) = "Inequalities = $(hypothesis_to_inequalities(x[1], x[2])), K = $(x[2])"

xlab              = "No. observations"
ylab_α            = "Proportion of errors"
ylab_α_familywise = "Familywise error rate"
ylab_β            = "Proportion of errors (β)"

plts_α = [
	make_figure(
		reduced_results_df[key],
		:α_error_prop_mean;
		xlabel = key[2] == 9 ? xlab : "",
		ylabel = key[1] === :p00 ? ylab_α : "",
		title  = key_to_title(key),
		legend = key[1] === :p00 && key[2] == 5 ? :topright : false,
		ylim   = (0, key[2] == 5 ? 0.4 : 1.05)
	)
	for key in keys_α
]

plts_α_familywise = [
	make_figure(
		reduced_results_df[key],
		:any_α_error_prop;
		xlabel = key[2] == 9 ? xlab : "",
		ylabel = key[1] === :p00 ? ylab_α_familywise : "",
		title  = key_to_title(key),
		legend = key[1] === :p00 && key[2] == 5 ? :topright : false,
		ylim   = (0, 1.05)
	)
	for key in keys_α
]

plts_β = [
	make_figure(
		reduced_results_df[key],
		:β_error_prop_mean;
		xlabel = key[2] == 9 ? xlab : "",
		ylabel = key[1] === :p25 ? ylab_β : "",
		title  = key_to_title(key),
		legend = key[1] === :p25 && key[2] == 5 ? :topright : false,
		ylim   = (0, 1.05),
	)
	for key in keys_β
]

plts_α_familywise_averaged = [make_figure(
	reduced_results_averaged_df[key],
	:any_α_error_prop_mean,
	xlabel = xlab,
	ylabel = "",#ylab_α_familywise,
	legend = false,
	title  = "Averaged",
	xlim   = (0, 550),
	xticks = [50, 100, 250, 500],
	yticks = key[1] == 5 ? (0:.1:.5) : :auto,
	ylim   = key[1] == 5 ? (0, .5) : :auto
) for key in keys(reduced_results_averaged_df)]

plts_β_averaged = [make_figure(
	reduced_results_averaged_df[key],
	:β_error_prop_mean_mean,
	xlabel = xlab,
	ylabel = "",#ylab_β,
	legend = false,
	title  = "Averaged",
	xlim   = (0, 550),
	xticks = [50, 100, 250, 500],
	ylim   = (0, .6),
	yticks = 0:.1:.6
) for key in keys(reduced_results_averaged_df)]

layout = (2, 4)
plt_α_joined = plot(
	plts_α..., layout = layout, size = 400 .* reverse(layout),
	bottom_margin = 8mm,
	left_margin = 12mm
)
layout = (2, 4)
plt_α_familywise_joined = plot(
	plts_α_familywise..., layout = layout, size = 400 .* reverse(layout),
	bottom_margin = 8mm,
	left_margin = 12mm
)
layout = (2, 4)
plt_β_joined = plot(
	plts_β..., layout = layout, size = 400 .* reverse(layout),
	bottom_margin = 8mm,
	left_margin = 12mm
)

# plots grouped by α, α-familywise, and β
# savefig(plt_α_joined,            joinpath("figures", "bigsimulation", "r100_alpha.pdf"))
# savefig(plt_α_familywise_joined, joinpath("figures", "bigsimulation", "r100_alpha_familywise.pdf"))
# savefig(plt_β_joined,            joinpath("figures", "bigsimulation", "r100_beta.pdf"))

# plots grouped by k
plts_k_5     = vcat(plts_α[1:4],            plts_β[1:4]);
plts_k_5_fam = vcat(plts_α_familywise[1:4], plts_β[1:4]);
plts_k_9     = vcat(plts_α[5:8],            plts_β[5:8]);
plts_k_9_fam = vcat(plts_α_familywise[5:8], plts_β[5:8]);

function update_titles!(plts, newtitles)
	for (plt, newtitle) in zip(plts, newtitles)
		plot!(plt; title = newtitle)
	end
end

plot!(plts_k_5[1], ylab = "α");
plot!(plts_k_5[5], legend = false);
plot!(plts_k_9[1], ylab = "α");
plot!(plts_k_9[5], legend = false);

plot!(plts_k_5_fam[5], legend = false);
plot!(plts_k_9_fam[5], legend = false);

newtitles_5 = ["Inequalities = " .* string.([0, 1, 2, 3]); "Inequalities = " .* string.([1, 2, 3, 4])]
newtitles_9 = ["Inequalities = " .* string.([0, 3, 5, 7]); "Inequalities = " .* string.([3, 5, 7, 8])]
update_titles!(plts_k_5,     newtitles_5)
update_titles!(plts_k_5_fam, newtitles_5)
update_titles!(plts_k_9,     newtitles_9)
update_titles!(plts_k_9_fam, newtitles_9)

neworder = [8, 5, 6, 7]
plts_k_5[5:8]     = plts_k_5[neworder]
plts_k_5_fam[5:8] = plts_k_5_fam[neworder]
plts_k_9[5:8]     = plts_k_9[neworder]
plts_k_9_fam[5:8] = plts_k_9_fam[neworder]
plot!(plts_k_5[5]    , ylab = ylab_β);
plot!(plts_k_5_fam[5], ylab = ylab_β);
plot!(plts_k_9[5]    , ylab = ylab_β);
plot!(plts_k_9_fam[5], ylab = ylab_β);

layout = (2, 4)
plt_k_5     = plot(plts_k_5...,     layout = layout, size = 400 .* reverse(layout), bottom_margin = 8mm, left_margin = 12mm);
plt_k_9     = plot(plts_k_9...,     layout = layout, size = 400 .* reverse(layout),	bottom_margin = 8mm, left_margin = 12mm);
plt_k_5_fam = plot(plts_k_5_fam..., layout = layout, size = 400 .* reverse(layout), bottom_margin = 8mm, left_margin = 12mm);
plt_k_9_fam = plot(plts_k_9_fam..., layout = layout, size = 400 .* reverse(layout),	bottom_margin = 8mm, left_margin = 12mm);

# savefig(plt_k_5,     joinpath("figures", "bigsimulation", "r100_groupedby_k_5.pdf"))
# savefig(plt_k_9,     joinpath("figures", "bigsimulation", "r100_groupedby_k_9.pdf"))
# savefig(plt_k_5_fam, joinpath("figures", "bigsimulation", "r100_groupedby_k_5_alpha_familywise.pdf"))
# savefig(plt_k_9_fam, joinpath("figures", "bigsimulation", "r100_groupedby_k_9_alpha_familywise.pdf"))

# new subset of plots
layout = @layout [
	[
		a; b
	] [
		c; d
	] [
		_; f{0.5h}; _
	]
]
# plts = collect(plot(randn(10)) for i in 1:5)
# plot(plts..., layout = layout)
ord  = [1, 3, 2, 4]
ordβ = [4, 2, 3, 1]
plts_α_familywise_5_averaged = [plts_α_familywise[ord];      plts_α_familywise_averaged[1]]
plts_α_familywise_9_averaged = [plts_α_familywise[ord .+ 4]; plts_α_familywise_averaged[2]]
plts_β_5_averaged = [plts_β[ordβ];      plts_β_averaged[1]]
plts_β_9_averaged = [plts_β[ordβ .+ 4]; plts_β_averaged[2]]
plot!(plts_α_familywise_5_averaged[2], xlab = xlab, ylab = ylab_α_familywise)
plot!(plts_α_familywise_9_averaged[1], xlab = "")
plot!(plts_α_familywise_9_averaged[3], xlab = "")
plot!(plts_α_familywise_5_averaged[4], xlab = xlab)
plot!(plts_α_familywise_9_averaged[2], xlab = xlab, ylab = ylab_α_familywise, legend = true)

# these two cut the uniform
plot!(plts_α_familywise_5_averaged[1], ylim = (0, .5), ytick = collect(0:.1:.5), legend = false);
plot!(plts_α_familywise_5_averaged[3], ylim = (0, .5), ytick = collect(0:.1:.5));
plot!(plts_α_familywise_5_averaged[2], legend = true);

plot!(plts_α_familywise_5_averaged[2], ylim = (0, .5), ytick = collect(0:.1:.5));
plot!(plts_α_familywise_5_averaged[4], ylim = (0, .5), ytick = collect(0:.1:.5));

plot!(plts_β_5_averaged[1], ylab = ylab_β, legend = true)
plot!(plts_β_5_averaged[2], ylab = ylab_β, legend = true, xlab = xlab)
plot!(plts_β_5_averaged[4], ylab = "",     legend = true, xlab = xlab)

plot!(plts_β_9_averaged[1], xlab = "",   ylab = ylab_β, legend = true)
plot!(plts_β_9_averaged[2], xlab = xlab, ylab = ylab_β)
plot!(plts_β_9_averaged[3], xlab = "",   ylab = "")
plot!(plts_β_9_averaged[4], xlab = xlab, ylab = "")

newtitles_β_5 = "Equalities = " .* string.([0, 2, 1, 3])
newtitles_β_9 = "Equalities = " .* string.([0, 5, 3, 7])

update_titles!(view(plts_β_5_averaged,            1:4), newtitles_β_5)
update_titles!(view(plts_β_9_averaged,            1:4), newtitles_β_9)

plt_α_familywise_5_joined = plot(plts_α_familywise_5_averaged..., layout = layout, size = 400 .* (3, 2), bottom_margin = 4mm, left_margin = 8mm)
plt_α_familywise_9_joined = plot(plts_α_familywise_9_averaged..., layout = layout, size = 400 .* (3, 2), bottom_margin = 4mm, left_margin = 8mm)
plt_β_5_joined            = plot(plts_β_5_averaged...,            layout = layout, size = 400 .* (3, 2), bottom_margin = 4mm, left_margin = 8mm)
plt_β_9_joined            = plot(plts_β_9_averaged...,            layout = layout, size = 400 .* (3, 2), bottom_margin = 4mm, left_margin = 8mm)

# savefig(plt_α_familywise_5_joined,     joinpath("figures", "bigsimulation", "subset_k_5_alpha_familywise.pdf"))
# savefig(plt_α_familywise_9_joined,     joinpath("figures", "bigsimulation", "subset_k_9_alpha_familywise.pdf"))
# savefig(plt_β_5_joined,                joinpath("figures", "bigsimulation", "subset_k_5_beta.pdf"))
# savefig(plt_β_9_joined,                joinpath("figures", "bigsimulation", "subset_k_9_beta.pdf"))


# subset of plots for manuscript
keys_α_subset = keys_α[[1, 2, 3]]
keys_β_subset = keys_β[[4, 1, 2]]

key_to_title2(x) = "Inequalities = $(hypothesis_to_inequalities(x[1], x[2]))"

plts_α_fam_subset = [
	make_figure(
		reduced_results_df[key],
		:any_α_error_prop;
		xlabel = "",
		ylabel = key[1] === :p00 ? "Probability of at least one error" : "",
		title  = key_to_title2(key),
		legend = key[1] === :p50 ? :topright : false,
		ylim   = (0, 0.5),
		xticks = [50, 100, 250, 500],
		xlim   = (0, 550)
	)
	for key in keys_α_subset
]

plts_β_subset = [
	make_figure(
		reduced_results_df[key],
		:β_error_prop_mean;
		xlabel = "no. observations",
		ylabel = key[1] === :p100 ? "Proportion of errors (β)" : "",
		title  = key_to_title2(key),
		legend = false,
		ylim   = (0, 0.75),
		xticks = [50, 100, 250, 500],
		xlim   = (0, 550)
	)
	for key in keys_β_subset
]

layout = (2, 3)
plt_k_5_subset = plot([plts_α_fam_subset; plts_β_subset]..., layout = layout, size = 400 .* reverse(layout), bottom_margin = 8mm, left_margin = 12mm);

savefig(plt_k_5_subset, joinpath("figures", "bigsimulation", "r100_groupedby_k_5_subset_2x3.pdf"))
