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
			CSV
import Turing.RandomMeasures: DirichletProcess

include("simulations/plotPartitions.jl")

updateDistribution(d::UniformMvUrnDistribution, args) = d
updateDistribution(d::BetaBinomialMvUrnDistribution, args) = BetaBinomialMvUrnDistribution(d.k, args...)
updateDistribution(d::RandomProcessMvUrnDistribution, args) = DirichletProcessMvUrnDistribution(d.k, args...)

make_title_wide(::UniformMvUrnDistribution) = "Uniform"
make_title_wide(d::BetaBinomialMvUrnDistribution) = "BetaBinomial α=$(d.α) β=$(d.β)"
make_title_wide(d::RandomProcessMvUrnDistribution) = "Dirichlet Process α=$(d.rpm.α)"

make_title(::Type{UniformMvUrnDistribution{Int64}}) = "Uniform"
make_title(::Type{BetaBinomialMvUrnDistribution{Int64}}) = "BetaBinomial"
make_title(::Type{RandomProcessMvUrnDistribution{DirichletProcess{Float64}, Int64}}) = "Dirichlet Process"

make_label(::Type{UniformMvUrnDistribution{Int64}}, args) = nothing
make_label(::Type{BetaBinomialMvUrnDistribution{Int64}}, args) = "α=$(args[1]) β=$(args[2])"
make_label(::Type{RandomProcessMvUrnDistribution{DirichletProcess{Float64}, Int64}}, args) = "α=$(args[1])"

function get_data(dists)

	k = length(dists[1][:dist])
	models = generate_distinct_models(k)

	models_int = parse.(Int, vec(mapslices(join, models, dims = 1)))
	df_wide_model_probs = DF.DataFrame(
		models = models_int
	)
	df_wide_incl_probs = DF.DataFrame(
		k = collect(0:k-1)
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
			incl_probs = logpdf_incl.(Ref(D), 0:k-1)

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

# use union of DPP & BetaBinomial
function make_all_plots(dfg, dfg_incl;
	inset_size = 0.125,
	graph_markersize = 4,
	graph_markerstroke = 1,
	ylims = (-9, 0),
	yticks = 0:-2:-8
)

	u = Set{Int}()
	for subdf in dfg
		new_idx = get_idx_unique_models(subdf)
		union!(u, Set(subdf[1, :models][new_idx]))
	end

	ord = sortperm(ordering.(digits.(values(u))), lt=!isless)
	x_models = collect(u)[ord]
	x_idx = [findfirst(==(model), first(dfg)[1, :models]) for model in x_models]

	x_axis_plots = plot_model.(x_models)
	for plt in x_axis_plots
		plot!(plt, size = (200, 200))
	end
	# x_axis_plots_data = plot_model_data.(x_models)

	# these determine
	# inset_size = 0.125
	# lim = 1.5
	# markersize = 5
	# markerstroke = 3

	plts = Matrix{Plots.Plot}(undef, 2, length(dfg))
	# (i, subdf) = first(enumerate(dfg))
	for (i, subdf) in enumerate(dfg)

		x = subdf[1, :models]
		y = Matrix{Float64}(undef, length(x_idx), size(subdf, 1))
		for j in axes(subdf, 1)
			y[:, j] .= subdf[j, :value][x_idx]
		end

		labels = reshape(make_label.(subdf[1, :distribution], subdf[!, :args]), (1, size(subdf, 1)))
		if subdf[1, :distribution]<: UniformMvUrnDistribution
			legendpos = :none
		else#if subdf[1, :distribution]<: BetaBinomialMvUrnDistribution
			legendpos = :topright
		# elseif subdf[1, :distribution]<: RandomProcessMvUrnDistribution
		# 	legendpos = :topright
		end

		plt0 = plot(eachindex(x_idx), y, markershape=:auto, title = make_title(subdf[1, :distribution]),
					legend = legendpos, labels = labels,
					ylims = ylims, yticks = yticks, xlims = (0, 8), xticks = 1:7)

		# (j, x_m) = first(enumerate(x_models))
		for (j, x_m) in enumerate(x_models)
			idx = j + 1
			x0 = (3 / 40) / (10 * inset_size) + 5*(j-1) / 40

			plot!(plt0; inset_subplots = [(1, bbox(x0, 0.05, inset_size, inset_size, :bottom, :left))],
				legend=false, border=:none, axis=nothing,
				subplot=idx
			);


			# plot_model!(plt0, x_m; #markersize = graph_markersize, markerstroke = graph_markerstroke,
			# 	inset_subplots = (1, bbox(x0, 0.05, inset_size, inset_size, :bottom, :left)),
			# 	legend=false, border=:none, axis=nothing,
			# 	subplot = idx
			# )


			# plot!(plt0, 0:1, 0:1,
			# 	inset_subplots = ((1, bbox(x0, 0.05, inset_size, inset_size, :bottom, :left))),
			# 	# legend=false, border=:none, axis=nothing,
			# 	subplot = idx
			# )

			# plot!(plt0, shape, alpha = .5, fillcolor = colors[count], linealpha = 0.0,
			# 	inset_subplots = (1, bbox(x0, 0.05, inset_size, inset_size, :bottom, :left)),
			# 	legend=false, border=:none, axis=nothing,
			# 	subplot = idx
			# )

			# plot_model!(plt0[idx], x_m)
			# plt0
			# plot!(plt0[idx], xlim = (-5, 5))

			plot_model!(plt0[idx], x_m; markersize = graph_markersize, markerstroke = graph_markerstroke)
			plot!(plt0[idx], #xlim = (-lim, lim), ylim = (-lim, lim),
				background_color_subplot = "transparent"
				# background_color_subplot = plot_color(:lightgrey, 0.15)
			)
		end
		# plt0

		plts[1, i] = plt0

		subdf_incl = dfg_incl[i]

		x = 0:4
		y = Matrix{Float64}(undef, length(x), size(subdf_incl, 1))
		for j in axes(subdf, 1)
			y[:, j] .= subdf_incl[j, :value]
		end

		# labels = reshape(make_label.(subdf_incl[1, :distribution], subdf_incl[!, :args]), (1, size(subdf_incl, 1)))
		# if subdf_incl[1, :distribution]<: UniformMvUrnDistribution
		# 	legendpos = :none
		# elseif subdf_incl[1, :distribution]<: BetaBinomialMvUrnDistribution
		# 	legendpos = :bottomleft
		# elseif subdf_incl[1, :distribution]<: RandomProcessMvUrnDistribution
		# 	legendpos = :bottomleft
		# end

		plt = plot(reverse(x), y, markershape = :auto,# labels = labels,
				legend = :none,#legendpos,
				ylims = ylims, yticks = yticks
		)

		plts[2, i] = plt

	end
	plts#, x_axis_plots
end

k=5
dists = (
	(
		dist = UniformMvUrnDistribution(k),
		args = ((k,),)
	),
	(
		dist = BetaBinomialMvUrnDistribution(k),
		args = ((1, 1), (k, 1))
	),
	(
		dist = DirichletProcessMvUrnDistribution(k, 1),
		args = ((0.5,), (1.0,), (Symbol("Gopalan Berry"),))
	)
)

df_wide_model_probs, df_long_model_probs, df_wide_incl_probs, df_long_incl_probs = get_data(dists);

# CSV.write(joinpath("tables", "model_probs_figure_2.csv"), df_wide_model_probs)
# CSV.write(joinpath("tables", "incl_probs_figure_2.csv"),  df_wide_incl_probs)

# TODO actually remake figure 2
dfg = DF.groupby(df_long_model_probs, :distribution);
dfg_incl = DF.groupby(df_long_incl_probs, :distribution);

plts = make_all_plots(dfg, dfg_incl);
ylabel!(plts[1, 1], "Prior probabilty");
ylabel!(plts[2, 1], "Prior probabilty");
xlabel!(plts[1, 2], "Model type");
xlabel!(plts[2, 2], "No. inequalities");



w = 600
jointplot = plot(permutedims(plts)..., layout = (2, 3), size = (3w, 2w));
savefig(jointplot, joinpath("figures", "visualizePriors_2x3.png"))
# savefig(jointplot, joinpath("figures", "visualizePriors_2x3.svg"))

# jointplot = plot(
# 	plts[1, :]...,
# 	deepcopy(x_axis)...,
# 	deepcopy(x_axis)...,
# 	deepcopy(x_axis)...,
# 	plts[2, :]...,
# 	layout = @layout [
# 		grid(1, 3,  heights = (0.7,  ))
# 		grid(1, 21, heights = (0.15, ))
# 		grid(1, 3,  heights = (0.7,  ))
# 	]
# )

# this is basically https://github.com/JuliaPlots/Plots.jl/issues/3378
# nr = 1
# nc = 1

# function show_plotmat(nr, nc)
# 	plotmat = [plot(randn(10), legend = false) for i in 1:nr, j in 1:nc]
# 	w = 600
# 	l = @layout grid(nr, nc)
# 	plot(plotmat..., layout = l, size = (nc * w, nr * w))
# 	# plot(plotmat..., layout = grid(nr, nc), size = (nc * w, nr * w))
# 	# plot(plotmat..., layout = (nr, nc), size = (nc * w, nr * w))
# end

# show_plotmat(1, 1) # good!
# show_plotmat(3, 3) # distances between axis tick labels and axis increases!
# show_plotmat(6, 2) # y-axis tick labels disappeared / fall outside of the plot area?
# show_plotmat(2, 6) # x-axis tick labels disappeared / fall outside of the plot area?

using Plots
using StatsPlots, StatsPlots.PlotMeasures
gr()
plot(contourf(randn(10, 20)), boxplot(rand(1:4, 1000), randn(1000)))

# Add a histogram inset on the heatmap.
# We set the (optional) position relative to bottom-right of the 1st subplot.
# The call is `bbox(x, y, width, height, origin...)`, where numbers are treated as
# "percent of parent".
histogram!(
    randn(1000),
    inset = (1, bbox(0.05, 0.05, 0.5, 0.25, :bottom, :right)),
    ticks = nothing,
    subplot = 3,
    bg_inside = nothing
)

# Add sticks floating in the window (inset relative to the window, as opposed to being
# relative to a subplot)
sticks!(
    randn(100),
    inset = bbox(0, -0.2, 200px, 100px, :center),
    ticks = nothing,
    subplot = 4
)

p0 = plot(1:10, 1:10);
plot!(p0, 0:1, 0:1,
	inset_subplots = (1, bbox(x0, 0.05, inset_size, inset_size, :bottom, :left)),
	subplot = 2
)

p0 = plot(1:10, 1:10);
plot!(p0, 0:1, 0:1,
	inset_subplots = (1, bbox(x0, 0.05, inset_size, inset_size, :bottom, :left)),
	subplot = 2
)
plot!(p0[2], xlim = (-3, 3))

p0 = plot(1:10, 1:10);
plot!(p0, 0:1, 0:1,
	inset_subplots = (1, bbox(x0, 0.05, inset_size, inset_size, :bottom, :left)),
	subplot = 2,
	xlim = (-3, 3) # works
);
p0

p1 = plot(1:10, 1:10);
plot!(p1, 0:1, 0:1,
	inset_subplots = (1, bbox(x0, 0.05, inset_size, inset_size, :bottom, :left)),
	subplot = 2
)
plot!(p1[2], xlim = (-3, 3));
p1


plot!(p0, 0:1, 0:1,
	inset_subplots = (1, bbox(x0, 0.05, inset_size, inset_size, :bottom, :left)),
	subplot = 2,
	xlim = (-3, 3)
)


