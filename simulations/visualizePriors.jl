using Base: Symbol
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

# Plots.PyPlotBackend()
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
function make_all_plots(dfg, dfg_incl)

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

	plts = Matrix{Plots.Plot}(undef, 2, length(dfg))
	# (i, subdf) = first(enumerate(dfg))
	for (i, subdf) in enumerate(dfg)

		x = subdf[1, :models]
		y = Matrix{Float64}(undef, length(x_idx), size(subdf, 1))
		for j in axes(subdf, 1)
			y[:, j] .= subdf[j, :value][x_idx]
		end

		plt0 = plot(eachindex(x_idx), y, markershape=:auto, title = make_title(subdf[1, :distribution]), legend = :none,
				ylims = (-7, 0), xlims = (0, 8), xticks = 1:7)

		inset_size = 0.1
		x0 = .8#1 / 40
		plot!(plt0; inset_subplots = [(1, bbox(x0, 0, inset_size, inset_size, :bottom, :left))], subplot=2, 
			legend=false, 
			# background_color_inside = plot_color(:lightgrey, 0.15),
			margin = 0.01mm,
			border=:none, axis=nothing
		)
		plot_model!(plt0[2], x_models[1])
		lim = 1.5
		plot!(plt0[2], xlim = (-lim, lim), ylim = (-lim, lim), background_color_subplot = "transparent")
		plt0
		# plot!(p[2], cos, fg_text=:white)
		plot_model!(plt0, x_models[1]; )


		graphplot(x_axis_plots_data[1][1], x = x_axis_plots_data[1][2], y = x_axis_plots_data[1][3])
				x_axis_plots_data[1]
		
		plot!(
			x_models[1];
			inset = (1, bbox(0.05, 0.05, 0.5, 0.25, :bottom, :right)),
			bg_inside = nothing
		)
		
		
		plot!(plt0, 

			# x_axis_plots[1],
			,
			inset = (1, bbox(0.0, -7, 1/7, 0.25, :bottom, :left)),
			bg_inside = nothing
		)

		l = @layout [
			a{0.8h}
			grid(1,length(x_axis_plots), heigths = (0.2, ))
		]
		# plts[1, i] = deepcopy(plot(plt0, deepcopy(x_axis_plots)..., layout = l))
		plts[1, i] = plot(plt0, deepcopy(x_axis_plots)..., layout = l)

		subdf_incl = dfg_incl[i]

		x = 0:4
		y = Matrix{Float64}(undef, length(x), size(subdf_incl, 1))
		for j in axes(subdf, 1)
			y[:, j] .= subdf_incl[j, :value]
		end

		labels = reshape(make_label.(subdf_incl[1, :distribution], subdf_incl[!, :args]), (1, size(subdf_incl, 1)))
		if subdf_incl[1, :distribution]<: UniformMvUrnDistribution
			legendpos = :none
		elseif subdf_incl[1, :distribution]<: BetaBinomialMvUrnDistribution
			legendpos = :bottomleft
		elseif subdf_incl[1, :distribution]<: RandomProcessMvUrnDistribution
			legendpos = :bottomleft
		end

		plt = plot(reverse(x), y, markershape = :auto, labels = labels, legend = legendpos, ylims = (-7, 0))

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




p0 = plot(heatmap(randn(10,20)), plot(rand(1:4,1000),randn(1000)), leg=false)
w = 1; h = 1
histogram!(p0, randn(1000), inset_subplots = [(1, bbox(0.05w,0.95h,0.5w,0.5h, v_anchor=:bottom))], subplot=1, ticks=nothing)
plot!(p0, p1)

sticks!(randn(100),subplot=4,inset_subplots=[bbox(0.35w,0.5h,200px,200px,h_anchor=:center,v_anchor=:center)])

x=range(0,2pi, length = 10)
y=range(0,2pi, length = 10)
z = [sin(xx)*cos(yy) for xx in x, yy in y]
p=contour(x,y,z,fill=true)
plot!(x, sin.(x),inset_subplots = [(1, bbox(0.05w,0.05h,0.3w,0.3h))], subplot=2)
plot!(p[2], cos, fg_text=:white)