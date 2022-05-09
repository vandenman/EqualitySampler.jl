#=

	This file visualizes the priors over partitions by their
		- probability of individual models
		- probability of including x (in)equalities

	Distributions:

		UniformConditionalUrnDistribution
		BetaBinomialConditionalUrnDistribution
		DirichletProcessDistribution (Chinese restaurant)

=#

using EqualitySampler, Distributions, Plots
import		DataFrames					as DF,
			OrderedCollections,
			CSV
import Turing.RandomMeasures: DirichletProcess
import  Plots.PlotMeasures: mm
include("priors_plot_colors_shapes_labels.jl")

import Printf
round_2_decimals(x::Number) = Printf.@sprintf "%.2f" x
round_2_decimals(x) = x

include("plot_partitions (Figure 1).jl")

updateDistribution(d::UniformMvUrnDistribution, args) = d
updateDistribution(d::BetaBinomialMvUrnDistribution, args) = BetaBinomialMvUrnDistribution(d.k, args...)
updateDistribution(d::RandomProcessMvUrnDistribution, args) = DirichletProcessMvUrnDistribution(d.k, args...)

make_title_wide(::UniformMvUrnDistribution) = "Uniform"
make_title_wide(d::BetaBinomialMvUrnDistribution) = "BetaBinomial α=$(d.α) β=$(d.β)"
make_title_wide(d::RandomProcessMvUrnDistribution) = "Dirichlet Process α=$(d.rpm.α)"

make_title(::Type{UniformMvUrnDistribution{Int64}}) = "Uniform prior"
make_title(::Type{BetaBinomialMvUrnDistribution{Int64}}) = "Beta-binomial prior"
make_title(::Type{RandomProcessMvUrnDistribution{DirichletProcess{Float64}, Int64}}) = "Dirichlet process prior"

make_label(::Type{UniformMvUrnDistribution{Int64}}, args) = nothing
make_label(::Type{BetaBinomialMvUrnDistribution{Int64}}, args) = "α=$(round_2_decimals(args[1])) β=$(round_2_decimals(args[2]))"
make_label(::Type{RandomProcessMvUrnDistribution{DirichletProcess{Float64}, Int64}}, args) = "α=$(round_2_decimals(args[1]))"

function get_data(k, priors)

	models = generate_distinct_models(k)

	models_int = parse.(Int, vec(mapslices(join, models, dims = 1)))
	df_wide_model_probs = DF.DataFrame(
		models = models_int
	)
	df_wide_incl_probs = DF.DataFrame(
		k = collect(1:k)
	)

	no_priors = length(priors)

	df_long_model_probs = DF.DataFrame(
		distribution	= Vector{DataType}(undef, no_priors),
		prior			= Vector{Symbol}(undef, no_priors),
		models			= Vector{Vector{Int}}(undef, no_priors),
		args			= Vector{String}(undef, no_priors),
		value			= Vector{Vector{Float64}}(undef, no_priors)
	)
	df_long_incl_probs = DF.DataFrame(
		distribution	= Vector{DataType}(undef, no_priors),
		prior			= Vector{Symbol}(undef, no_priors),
		models			= Vector{Vector{Int}}(undef, no_priors),
		args			= Vector{String}(undef, no_priors),
		value			= Vector{Vector{Float64}}(undef, no_priors)
	)

	i = 1
	for prior in priors

		D = instantiate_prior(prior, k)
		model_probs = vec(mapslices(m->logpdf_model_distinct(D, m), models, dims = 1))
		incl_probs = logpdf_incl.(Ref(D), 1:k)

		nm = make_title_wide(D)
		df_wide_model_probs[!, nm] = model_probs
		df_wide_incl_probs[!, nm] = incl_probs

		row = (
			distribution = typeof(D),
			prior = prior,
			models = models_int,
			args = get_args_from_prior(prior),
			value = model_probs
		)
		df_long_model_probs[i, :] = row

		row = (
			distribution = typeof(D),
			prior = prior,
			models = models_int,
			args = get_args_from_prior(prior),
			value = incl_probs
		)
		df_long_incl_probs[i, :] = row

		i+=1
	end
	return df_wide_model_probs, df_long_model_probs, df_wide_incl_probs, df_long_incl_probs
end

function get_data(dists)

	k = length(dists[1][:dist])
	models = generate_distinct_models(k)

	models_int = parse.(Int, vec(mapslices(join, models, dims = 1)))
	df_wide_model_probs = DF.DataFrame(
		models = models_int
	)
	df_wide_incl_probs = DF.DataFrame(
		k = collect(1:k)
	)

	ndists = sum(max(1, length(elem[:args])) for elem in dists)

	df_long_model_probs = DF.DataFrame(
		distribution	= Vector{DataType}(undef, ndists),
		models			= Vector{Vector{Int}}(undef, ndists),
		args			= Vector{Tuple}(undef, ndists),
		value			= Vector{Vector{Float64}}(undef, ndists)
	)
	df_long_incl_probs = DF.DataFrame(
		distribution	= Vector{DataType}(undef, ndists),
		models			= Vector{Vector{Int}}(undef, ndists),
		args			= Vector{Tuple}(undef, ndists),
		value			= Vector{Vector{Float64}}(undef, ndists)
	)

	i = 1
	for (d, args) in dists
		@show d, args, i
		for arg in args
			D = updateDistribution(d, arg)
			model_probs = vec(mapslices(m->logpdf_model_distinct(D, m), models, dims = 1))
			incl_probs = logpdf_incl.(Ref(D), 1:k)

			nm = make_title_wide(D)
			df_wide_model_probs[!, nm] = model_probs
			df_wide_incl_probs[!, nm] = incl_probs

			row = (
				distribution = typeof(D),
				models = models_int,
				args = arg,
				value = model_probs
			)
			df_long_model_probs[i, :] = row

			row = (
				distribution = typeof(D),
				models = models_int,
				args = arg,
				value = incl_probs
			)
			df_long_incl_probs[i, :] = row

			i+=1
		end
	end
	return df_wide_model_probs, df_long_model_probs, df_wide_incl_probs, df_long_incl_probs
end

function get_idx_unique_models(subdf)
	tmp = unique(x->round(x, sigdigits = 3), subdf[1, :value])
	return [findfirst(==(tmp[j]), subdf[1, :value]) for j in eachindex(tmp)]
end

function make_all_plots(dfg, dfg_incl;
	inset_size = 0.125,
	graph_markersize = 4,
	partition_markersize = 4,
	graph_markerstroke = 1,
	ylims = (-9, 0),
	yticks = 0:-2:-8,
	markersize = 7,
	markerstrokewidth = 0.9
)

	u = Set{Int}()
	for subdf in dfg
		new_idx = get_idx_unique_models(subdf)
		union!(u, Set(subdf[1, :models][new_idx]))
	end

	ord = sortperm(ordering.(digits.(values(u))), lt=!isless)
	x_models = collect(u)[ord]
	x_idx = [findlast(==(model), first(dfg)[1, :models]) for model in x_models]

	x_axis_plots = plot_model.(x_models)
	for plt in x_axis_plots
		plot!(plt, size = (200, 200), markersize = partition_markersize)
	end

	plts = Matrix{Plots.Plot}(undef, 2, length(dfg))
	# (i, subdf) = first(enumerate(dfg))
	for (i, subdf) in enumerate(dfg)

		x = subdf[1, :models]
		y = Matrix{Float64}(undef, length(x_idx), size(subdf, 1))
		for j in axes(subdf, 1)
			y[:, j] .= subdf[j, :value][x_idx]
		end

		labels = get_labels(subdf[!, :prior])#reshape(make_label.(subdf[1, :distribution], subdf[!, :args]), (1, size(subdf, 1)))
		colors = get_colors(subdf[!, :prior], 0.95)
		shapes = get_shapes(subdf[!, :prior])



		if subdf[1, :distribution]<: UniformMvUrnDistribution
			legendpos = :none
		else
			legendpos = :topright
			if subdf[1, :distribution]<: BetaBinomialMvUrnDistribution
				start = 4
			else
				start = 5
			end
			for i in eachindex(labels)
				labels[i] = labels[i][start:end]
			end
		end

		if subdf[1, :distribution] <: RandomProcessMvUrnDistribution{DirichletProcess{Float64}, Int64}
			y = y[:, [2, 1, 3]]
			labels = labels[:, [2, 1, 3]]
		end

		plt0 = plot(repeat(eachindex(x_idx), size(y, 2)), vec(y), group = repeat(axes(y, 2), inner = size(y, 1)),
			markershapes = repeat(shapes, inner = length(x_idx)),
			color = repeat(colors, inner = length(x_idx)),
			title = make_title(subdf[1, :distribution]),
			legend = legendpos, labels = labels,
			markersize = markersize,
			markerstrokewidth = markerstrokewidth,
			ylims = ylims, yticks = yticks, xlims = (0, 8), xticks = (1:7, fill("", 7)))

		for (j, x_m) in enumerate(x_models)
			idx = j + 1
			x0 = (3 / 40) / (10 * inset_size) + 5*(j-1) / 40

			plot!(plt0; inset_subplots = [(1, bbox(x0, 0.05, inset_size, inset_size, :bottom, :left))],
				legend=false, border=:none, axis=nothing,
				subplot=idx
			);

			plot_model!(plt0[idx], x_m; markersize = graph_markersize, markerstroke = graph_markerstroke)
			plot!(plt0[idx], #xlim = (-lim, lim), ylim = (-lim, lim),
				background_color_subplot = "transparent"
				# background_color_subplot = plot_color(:lightgrey, 0.15)
			)
		end

		plts[1, i] = plt0

		subdf_incl = dfg_incl[i]

		k = ndigits(first(first(subdf.models)))
		x = 0:k-1 # no. inequalities
		# x = 1:k # no. free parameters

		y = Matrix{Float64}(undef, length(x), size(subdf_incl, 1))
		for j in axes(subdf, 1)
			y[:, j] .= subdf_incl[j, :value]
		end

		if subdf[1, :distribution] <: RandomProcessMvUrnDistribution{DirichletProcess{Float64}, Int64}
			y = y[:, [2, 1, 3]]
		end

		plt = plot(repeat(eachindex(x), size(y, 2)), vec(y), group = repeat(axes(y, 2), inner = size(y, 1)),
				markershape = repeat(shapes, inner = length(x)),
				color = repeat(colors, inner = length(x)),
				legend = :none,
				ylims = ylims, yticks = yticks,
				markerstrokewidth = markerstrokewidth,
				markersize = markersize
		)

		plts[2, i] = plt

	end
	plts#, x_axis_plots
end

k = 5
priors = (
	:uniform,
	:BetaBinomial11,
	:BetaBinomial1k,
	:BetaBinomial1binomk2,
	:DirichletProcess0_5,
	:DirichletProcess1_0,
	:DirichletProcessGP
)
df_wide_model_probs, df_long_model_probs, df_wide_incl_probs, df_long_incl_probs = get_data(k, priors)

# CSV.write(joinpath("tables", "model_probs_figure_2.csv"), df_wide_model_probs)
# CSV.write(joinpath("tables", "incl_probs_figure_2.csv"),  df_wide_incl_probs)

dfg = DF.groupby(df_long_model_probs, :distribution);
dfg_incl = DF.groupby(df_long_incl_probs, :distribution);

plts = make_all_plots(dfg, dfg_incl; graph_markersize = 5, markersize = 9, markerstrokewidth = 0.5);
plts = plts[:, [3, 2, 1]];
ylabel!(plts[1, 1], "Log prior probabilty");
ylabel!(plts[2, 1], "Log prior probabilty");
xlabel!(plts[1, 2], "Model type");
xlabel!(plts[2, 2], "No. inequalities");
# xlabel!(plts[2, 2], "No. free parameters");
plot!(plts[1, 1]; foreground_color_legend = nothing, background_color_legend = nothing);
plot!(plts[1, 2]; foreground_color_legend = nothing, background_color_legend = nothing);
# plot!(plts[2, 2], bottom_margin = 10mm);
# plot!(plts[1, 1], left_margin = 15mm);
# plot!(plts[2, 1], left_margin = 15mm);

w = 600
joint_plts = permutedims(plts)
Plots.resetfontsizes()
Plots.scalefontsizes(1.8) # only run this once!
jointplot = plot(joint_plts..., legendfont = font(12), titlefont = font(24), layout = (2, 3), size = (3w, 2w),
				bottom_margin = 10mm, left_margin = 15mm);
savefig(jointplot, joinpath("figures", "visualizePriors_2x3_new.pdf"))
