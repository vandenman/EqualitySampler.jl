using Plots, Plots.PlotMeasures, DataFrames, Chain, Colors, ColorSchemes, Printf
using Statistics

include("simulation_helpers.jl")

results_dir = joinpath("simulations", "small_simulation_runs")
results_df = read_results(results_dir)

# import TableView, Blink
# TableView.showtable(results_df)

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
		:Westfall_uncorrected	=> "Westfall_U",
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

function make_figure(df, x_symbol, y_symbol; kwargs...)
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
		df[!, x_symbol] .+ offset,
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
p_null_prop2  = make_figure(reduced_results_df[(hypothesis=:p00,)],  :groups, :prop_incorrect_mean; xlabel = "",           ylabel = "Proportion of errors", title = "Null model",     legend = :topleft);
p_full_prop2  = make_figure(reduced_results_df[(hypothesis=:p100,)], :groups, :prop_incorrect_mean; xlabel = "",           ylabel = "",                     title = "Full model",     legend = false);
p_lambda_0_50 = make_figure(lambda_results_df,                       :groups, :lambda_0_50_mean;    xlabel = "No. groups", ylabel = "",                     title = make_title(0.50), legend = false,   ylim = (0, 0.6), yticks = 0:0.1:0.6);

joined_plot_lambda = plot(
	p_null_prop2,
	plot(p_full_prop2, xlab = "Number of groups"),
	plot(p_lambda_0_50, xlab = "", title = "Null model + Full model"),
	layout = (1, 3),
	left_margin = 7mm,
	bottom_margin = 7mm
);

savefig(plot(joined_plot_lambda, size = (3*450, 450)), joinpath("figures", "smallsimulation_rep_20_partial.pdf"))