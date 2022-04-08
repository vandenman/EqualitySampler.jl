using Plots, Plots.PlotMeasures, DataFrames, Chain, Colors, ColorSchemes, Printf

include("simulation_helpers.jl")

results_dir = joinpath("simulations", "big_simulation_runs")
results_df = read_results(results_dir)


priors_to_remove = Set((:BetaBinomialk1, #=:DirichletProcessGP,=# :DirichletProcess2_0))
reduced_results_df = @chain results_df begin
	filter(:prior => x -> x ∉ priors_to_remove, _)
	groupby(Cols(:obs_per_group, :prior, :groups, :hypothesis))
	combine(:any_incorrect => mean, :prop_incorrect => mean)
	sort([order(:hypothesis, by=x->parse(Int, string(x)[2:end]), rev=true), order(:obs_per_group)])
	groupby(Cols(:hypothesis, :groups))
end

lambda_results_df = @chain results_df begin
	filter(:prior => x -> x ∉ priors_to_remove, _)
	unstack([:obs_per_group, :repeat, :groups, :prior], :hypothesis, :prop_incorrect)
	# transform(
	# 	[:null, :full] => ((n, f) -> 0.50 * n + 0.50 * f) => :lambda_0_50,
	# 	[:null, :full] => ((n, f) -> 0.95 * n + 0.05 * f) => :lambda_0_95
	# )
	groupby(Cols(:obs_per_group, :prior, :groups))
	combine([:p00, :p25, :p50, :p75, :p100] .=> mean)
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

function make_figure(df, y_symbol; kwargs...)
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

		ylim   = (0, 1.05),
		yticks = 0:.2:1,
		xticks = [250, 500, 750, 1000],
		xlim   = (200, 1050),
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
plt_prop = [
	make_figure(
		reduced_results_df[key],
		:prop_incorrect_mean;
		# xlabel = "no. observations",
		xlabel = key[2] == 9 ? "no. observations" : "",
		ylabel = key[1] === :p00 ? "Proportion of errors" : "",
		title = join(key, "-"),
		legend = key[1] === :p100 && key[2] == 5 ? :topleft : false
	)
	for key in keys_ordered
]

# layout = (1, 5)
layout = (2, 5)
plt_joined = plot(
	plt_prop..., layout = layout, size = 400 .* reverse(layout),
	bottom_margin = 8mm,
	left_margin = 12mm
)
savefig(plt_joined, joinpath("figures", "bigsimulation_rep_10_initialplot.pdf"))