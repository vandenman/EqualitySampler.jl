using Plots, Plots.PlotMeasures, DataFrames, Chain, Colors, ColorSchemes, Printf
using Statistics

include("priors_plot_colors_shapes_labels.jl")
include("simulation_helpers.jl")
results_joined_path = joinpath("simulations", "small_simulation_runs_joined", "small_simulation_runs_joined.jld2")
if !isfile(results_joined_path)
	results_dir = joinpath("simulations", "small_simulation_runs")
	results_df = read_results(results_dir)
	JLD2.jldsave(results_joined_path; results_df = results_df)
else
	tmp = JLD2.jldopen(results_joined_path)
	results_df = tmp["results_df"]
end

priors_to_remove = Set((:BetaBinomialk1, #=:DirichletProcessGP,=# :DirichletProcess2_0))
function recode_hypothesis(x)
	for i in eachindex(x)
		if x[i] === :null
			x[i] = :p00
		elseif x[i] === :full
			x[i] = :p100
		end
	end
	x
end
reduced_results_df = @chain results_df begin
	filter(:prior => x -> x ∉ priors_to_remove, _)
	transform(:hypothesis => recode_hypothesis => :hypothesis)
	groupby(Cols(:obs_per_group, :prior, :groups, :hypothesis))
	combine([:any_incorrect, :prop_incorrect, :α_error_prop, :β_error_prop] .=> mean)
	sort([order(:hypothesis, by=x->parse(Int, string(x)[2:end]), rev=true), order(:groups)])
	groupby(:hypothesis)
end

lambda_results_df = @chain results_df begin
	filter(:prior => x -> x ∉ priors_to_remove, _)
	transform(:hypothesis => recode_hypothesis => :hypothesis)
	unstack([:obs_per_group, :repeat, :groups, :prior], :hypothesis, :prop_incorrect)
	transform(
		[:p00, :p100] => ((n, f) -> 0.50 * n + 0.50 * f) => :lambda_0_50
	)
	groupby(Cols(:prior, :groups))
	combine(:lambda_0_50 => mean)
	sort([order(:groups)])
end

function make_figure_small(df, x_symbol, y_symbol; kwargs...)
	# colors1 = get_colors(df[!, :prior], .8)

	idx_bb1k   = findall(x->x===:BetaBinomial1k, df[!, :prior])
	idx_bb1bk2 = findall(x->x===:BetaBinomial1binomk2, df[!, :prior])

	oo = collect(axes(df, 1))
	oo[idx_bb1k], oo[idx_bb1bk2] = oo[idx_bb1bk2], oo[idx_bb1k]
	prior = df[!, :prior]

	colors2 = get_colors(prior, .5)
	shapes  = get_shapes(prior)
	labels  = get_labels(prior)

	labels[2], labels[3] = labels[3], labels[2]

	offset_size = 0.1
	offset_lookup = Dict(
		((:BetaBinomial11, :BetaBinomialk1, :BetaBinomial1k, :BetaBinomial1binomk2) .=> -offset_size)...,
		((:DirichletProcess0_5, :DirichletProcess1_0, :DirichletProcess2_0, :DirichletProcessGP) .=> offset_size)...,
		((:uniform, :Westfall, :Westfall_uncorrected) .=> 0.0)...
	)
	offset = [offset_lookup[p] for p in prior]
	Plots.plot(
		df[!, x_symbol] .+ offset,
		df[!, y_symbol],
		group = df[oo, :prior],

		linecolor			= colors2,
		markercolor			= colors2,
		markershape			= shapes,
		# markerstrokealpha	= 0.0,
		markerstrokewidth	= 0,
		# markerstrokecolor	= colors1,
		labels				= labels,
		markersize			= 7,
		markeralpha			= 0.75,

		# ylim   = (0, 1.05),
		# yticks = 0:.2:1,
		# xticks = [250, 500, 750, 1000],
		# xlim   = (200, 1050),
		xlim   = (1, 11),

		background_color_legend = nothing,
		foreground_color_legend = nothing
		;
		kwargs...
	)
end

make_title(λ) = @sprintf "%.2f * (errors | null model) +\n%.2f * (errors | full model)    " λ 1 - λ
p_null_prop2  = make_figure_small(reduced_results_df[(hypothesis=:p00,)],  :groups, :prop_incorrect_mean; xlabel = "",           ylabel = "Proportion of errors", title = "Null model",     legend = :topleft);
p_full_prop2  = make_figure_small(reduced_results_df[(hypothesis=:p100,)], :groups, :prop_incorrect_mean; xlabel = "",           ylabel = "",                     title = "Full model",     legend = false);
p_lambda_0_50 = make_figure_small(lambda_results_df,                       :groups, :lambda_0_50_mean;    xlabel = "No. groups", ylabel = "",                     title = make_title(0.50), legend = false,   ylim = (0, 0.6), yticks = 0:0.1:0.6);

joined_plot_lambda = plot(
	p_null_prop2,
	plot(p_full_prop2, xlab = "Number of groups"),
	plot(p_lambda_0_50, xlab = "", title = "Null model + Full model"),
	layout = (1, 3),
	left_margin = 7mm,
	bottom_margin = 7mm
);

# savefig(plot(joined_plot_lambda, size = (3*450, 450)), joinpath("figures", "smallsimulation", "3_panel_alpha.pdf"))

p_null_any_error = make_figure_small(reduced_results_df[(hypothesis=:p00,)],  :groups, :any_incorrect_mean;  xlabel = "No. groups", ylabel = "Probability of at least one error", title = "Null model",  legend = :topleft)
p_full_any_error = make_figure_small(reduced_results_df[(hypothesis=:p100,)], :groups, :prop_incorrect_mean; xlabel = "No. groups", ylabel = "Proportion of errors (β)",          title = "Full model",  legend = false)

joined_plot_familywise_error = plot(
	p_null_any_error,
	p_full_any_error,
	layout = (1, 2),
	left_margin = 7mm,
	bottom_margin = 7mm
);
savefig(plot(joined_plot_familywise_error, size = (2*500, 500)), joinpath("figures", "smallsimulation", "2_panel_alpha_familywise.pdf"))
