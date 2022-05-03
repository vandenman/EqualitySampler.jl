using Plots, Plots.PlotMeasures, DataFrames, Chain, Colors, ColorSchemes, Printf
using Statistics

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
	filter(:prior => x -> x ∉ priors_to_remove, _)
	# filter(:obs_per_group => x-> x in obs_per_group_to_keep, _)
	groupby(Cols(:obs_per_group, :prior, :groups, :hypothesis))
	# combine(:any_incorrect => mean, :prop_incorrect => mean)
	combine([:any_incorrect, :prop_incorrect, :α_error_prop, :β_error_prop] .=> mean, :α_error_prop => (x->mean(>(0.0), x)) => :any_α_error_prop)
	sort([order(:hypothesis, by=x->parse(Int, string(x)[2:end]), rev=true), order(:obs_per_group)])
end
reduced_results_df = groupby(reduced_results_df_ungrouped, Cols(:hypothesis, :groups))

reduced_results_df2 = @chain reduced_results_df_ungrouped begin
	filter(:obs_per_group => x-> x in obs_per_group_to_keep, _)
	groupby(Cols(:hypothesis, :groups))
end

function get_labels(priors)
	lookup = Dict(
		:uniform				=> "Uniform",
		:BetaBinomial11			=> "BB α=1, β=1",
		:BetaBinomialk1			=> "BB α=K, β=1",
		:BetaBinomial1k			=> "BB α=1, β=K",
		:BetaBinomial1binomk2	=> "BB α=1, β=binom(K,2)",
		# :BetaBinomial1binomk2	=> L"\mathrm{BB}\,\,\alpha=K, \beta=\binom{K}{2}",
		:DirichletProcess0_5	=> "DPP α=0.5",
		:DirichletProcess1_0	=> "DPP α=1",
		:DirichletProcess2_0	=> "DPP α=2",
		# :DirichletProcessGP		=> "DPP α=Gopalan & Berry",
		:DirichletProcessGP		=> "DPP α=G&B",
		:Westfall				=> "Westfall",
		:Westfall_uncorrected	=> "Pairwise BFs",
	)
	priors_set = sort!(unique(priors))
	return reshape([lookup[prior] for prior in priors_set], 1, length(priors_set))
end

function get_colors(priors, alpha = 0.75)
	colors = alphacolor.(ColorSchemes.seaborn_colorblind[1:10], alpha)
	lookup = Dict(
		:uniform              => colors[1],
		:BetaBinomial11       => colors[2],
		:BetaBinomialk1       => colors[3],
		:BetaBinomial1k       => colors[4],
		:BetaBinomial1binomk2 => colors[5],
		:DirichletProcess0_5  => colors[6],
		:DirichletProcess1_0  => colors[7],
		:DirichletProcess2_0  => colors[8],
		:DirichletProcessGP   => colors[9],
		:Westfall             => colors[10],
		:Westfall_uncorrected => colors[3]
	)
	return [lookup[prior] for prior in priors]
end

function get_shapes(priors)
	lookup = Dict(
		:uniform              => :rect,
		:BetaBinomial11       => :utriangle,
		:BetaBinomialk1       => :rtriangle,
		:BetaBinomial1k       => :ltriangle,
		:BetaBinomial1binomk2 => :dtriangle,
		:DirichletProcess0_5  => :star4,
		:DirichletProcess1_0  => :star5,
		:DirichletProcess2_0  => :star6,
		:DirichletProcessGP   => :star8,
		:Westfall             => :circle,
		:Westfall_uncorrected => :circle
	)
	return [lookup[prior] for prior in priors]
end

function make_figure(df, y_symbol; xticks = [50, 100, 250, 500, 750, 1000], xlim = (0, 1050), kwargs...)
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
keys_α  = filter(x->x.hypothesis != :p100, keys_ordered)
keys_β  = filter(x->x.hypothesis != :p00,  keys_ordered)
key_to_title(x) = "Inequalities = $(hypothesis_to_inequalities(x[1], x[2])), K = $(x[2])"

plts_α = [
	make_figure(
		reduced_results_df[key],
		:α_error_prop_mean;
		xlabel = key[2] == 9 ? "no. observations" : "",
		ylabel = key[1] === :p00 ? "Proportion of errors" : "",
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
		xlabel = key[2] == 9 ? "no. observations" : "",
		ylabel = key[1] === :p00 ? "Probability of at least one error" : "",
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
		xlabel = key[2] == 9 ? "no. observations" : "",
		ylabel = key[1] === :p25 ? "Proportion of errors (β)" : "",
		title  = key_to_title(key),
		legend = key[1] === :p25 && key[2] == 5 ? :topright : false,
		ylim   = (0, 1.05),
	)
	for key in keys_β
]

# layout = (1, 5)
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
plt_β_joined = plot(
	plts_β..., layout = layout, size = 400 .* reverse(layout),
	bottom_margin = 8mm,
	left_margin = 12mm
)

# plots grouped by α, α-familywise, and β
savefig(plt_α_joined,            joinpath("figures", "bigsimulation", "r100_alpha.pdf"))
savefig(plt_α_familywise_joined, joinpath("figures", "bigsimulation", "r100_alpha_familywise.pdf"))
savefig(plt_β_joined,            joinpath("figures", "bigsimulation", "r100_beta.pdf"))

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

newtitles_5 = ["Inequalities=" .* string.([0, 2, 3, 4]); "Inequalities=" .* string.([2, 3, 4, 5])]
newtitles_9 = ["Inequalities=" .* string.([0, 3, 5, 7]); "Inequalities=" .* string.([3, 5, 7, 9])]
update_titles!(plts_k_5,     newtitles_5)
update_titles!(plts_k_5_fam, newtitles_5)
update_titles!(plts_k_9,     newtitles_9)
update_titles!(plts_k_9_fam, newtitles_9)

neworder = [8, 5, 6, 7]
plts_k_5[5:8]     = plts_k_5[neworder]
plts_k_5_fam[5:8] = plts_k_5_fam[neworder]
plts_k_9[5:8]     = plts_k_9[neworder]
plts_k_9_fam[5:8] = plts_k_9_fam[neworder]
plot!(plts_k_5[5]    , ylab = "Proportion of errors (β)");
plot!(plts_k_5_fam[5], ylab = "Proportion of errors (β)");
plot!(plts_k_9[5]    , ylab = "Proportion of errors (β)");
plot!(plts_k_9_fam[5], ylab = "Proportion of errors (β)");


plt_k_5     = plot(plts_k_5...,     layout = layout, size = 400 .* reverse(layout), bottom_margin = 8mm, left_margin = 12mm);
plt_k_9     = plot(plts_k_9...,     layout = layout, size = 400 .* reverse(layout),	bottom_margin = 8mm, left_margin = 12mm);
plt_k_5_fam = plot(plts_k_5_fam..., layout = layout, size = 400 .* reverse(layout), bottom_margin = 8mm, left_margin = 12mm);
plt_k_9_fam = plot(plts_k_9_fam..., layout = layout, size = 400 .* reverse(layout),	bottom_margin = 8mm, left_margin = 12mm);

savefig(plt_k_5, joinpath("figures", "bigsimulation", "r100_groupedby_k_5.pdf"))
savefig(plt_k_9, joinpath("figures", "bigsimulation", "r100_groupedby_k_9.pdf"))
savefig(plt_k_5_fam, joinpath("figures", "bigsimulation", "r100_groupedby_k_5_alpha_familywise.pdf"))
savefig(plt_k_9_fam, joinpath("figures", "bigsimulation", "r100_groupedby_k_9_alpha_familywise.pdf"))

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
