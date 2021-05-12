#=

	This file visualizes the priors over partitions by their
		- probability of individual models
		- probability of including x (in)equalities

	Distributions:

		UniformConditionalUrnDistribution
		BetaBinomialConditionalUrnDistribution
		DirichletProcessDistribution (Chinese restaurant)

=#

using EqualitySampler, Plots, Turing
import StatsBase: countmap

# include("simulations/samplePriorsTuring.jl")
# include("simulations/plotFunctions.jl")

function linear_interpolation(new_x, known_y::AbstractVector{T}, known_x) where T
	# assumes new_x and known_x is sorted from small to large
	!issorted(new_x)	&& throw(DomainError(new_x,		"new_x should be sorted"))
	!issorted(known_x)	&& throw(DomainError(known_x,	"known_x should be sorted"))

	result = Vector{T}(undef, length(new_x))
	j0 = 1
	for i in eachindex(new_x)

		# find the index of the largest value in new_x smaller than new_x[i]
		for j in j0:length(known_x)
			if known_x[j] > new_x[i]
				j0 = j - 1
				break
			end
		end

		if j0 == length(known_x)
			i1 = j0 - 1
			i2 = j0
		else
			i1 = j0
			i2 = i1 + 1
		end

		w = (new_x[i] - known_x[i1]) / (known_x[i2] - known_x[i1])
		result[i] = known_y[i1] * (1 - w) + known_y[i2] * w

	end
	result
end

function ordering(x)
	# TODO: ensure this is the actual ordering like on wikipedia
	d = countmap(x)
	res = Float64(length(x) - length(d))

	v = sort!(collect(values(d)), lt = !isless)
	# res = 0.0
	for i in eachindex(v)
		res += v[i] ./ 10 .^ (i)
	end
	return res
end

function empty_plot(k; yaxis = :log, legend = true, xrotation = 45, xfontsize = 8, palette = :default)

	allmodels = generate_distinct_models(k)
	order = sortperm(ordering.(eachcol(allmodels)), lt = !isless)
	allmodels_str = [join(col) for col in eachcol(view(allmodels, :, order))]
	transparent = Colors.RGBA(0, 0, 0, 0)
	plt = plot(allmodels_str, ones(length(allmodels_str)), label = nothing, color = transparent,
				yaxis = yaxis, legend = legend, xrotation = xrotation,
				xticks = (0.5:length(allmodels_str)-.5, allmodels_str),
				xtickfont=font(xfontsize),
				tickfontvalign = :bottom,
				palette = palette)
	plt
end

function model_pmf!(plt, D, colors = 1:k, labels = :none)

	k = length(D)
	y = expected_model_probabilities(D)
	incl_size = expected_inclusion_counts(k)
	cumulative_sum = 0
	xstart = 0

	if labels === :none || length(labels) != k
		labs = incl_size
	else
		labs = labels
	end

	for i in 1:k
		yval = y[cumulative_sum + incl_size[i]]
		ycoords = [yval, yval]
		if isone(incl_size[i])
			xcoords = [xstart + 0.5, xstart + 0.5]
			scatter!(plt, xcoords, ycoords, m = 4, label = string(labs[i]), #yaxis = yaxis,
					 color = colors[i], markerstrokecolor = colors[i]);
		else
			xcoords = [xstart, xstart + incl_size[i]]
			plot!(plt, xcoords, ycoords, lw = 4, label = string(labs[i]), #yaxis = yaxis,
				  color = colors[i]);
		end
		cumulative_sum += incl_size[i]
		xstart += incl_size[i]
	end
	return plt
end

function incl_pmf!(plt, D; yaxis = :log, palette = :default, color = 1:length(D), label = fill("", length(D)))
	k = length(D)
	incl_probs = expected_inclusion_probabilities(D)
	scatter!(plt, 0:k-1, incl_probs, m = 4, yaxis = yaxis, legend = false,
				  color = color, markerstrokecolor = color, palette = palette, label = label);

	xx = [i + j for i in 0:k-2 for j in (0.05, 0.95)]
	yy = linear_interpolation(xx, incl_probs, collect(0:k-1))
	xm = reshape(xx, (2, k-1))
	ym = reshape(yy, (2, k-1))
	plot!(plt, xm, ym, color = color, palette = palette, label = label, yaxis = yaxis, legend = false)
	# plot!(plt, 0:k-1, incl_probs, color = color, palette = palette, label = label, yaxis = yaxis, legend = false)
end

k = 5
Duniform = UniformConditionalUrnDistribution(ones(Int, k))
DBetaBinom = BetaBinomialConditionalUrnDistribution(ones(Int, k), 1, 1, 1)
DDirichletProcess = RandomProcessDistribution(Turing.RandomMeasures.DirichletProcess(1.887), k)

# TODO: for some odd reason the x-axis tick labels are offset...
yaxis_scale = :none
xrotation   = 90
xfontsize   = 4
ylimsModel  = (0, .25)
ylimsIncl   = (0, .5)
colorpalet  = :seaborn_colorblind
palette(colorpalet);

dists = (
	(
		dist = UniformMvUrnDistribution(k)
	),
	(
		dist = BetaBinomialMvUrnDistribution(k),
		args = ((.75, 1.5), (1, 1), (.5, .5))
	),
	(
		dist = RandomProcessDistribution(Turing.RandomMeasures.DirichletProcess(1.887), k),
		dist = (1.887, 1.0, 3.0)
	)
)

# TODO: this could just be a loop over distributions with parameter vectors
# αβ = ((.5, 2), (2, 2), (1, 1), (.5, .5)) # Beta Binomial parameters
αβ = ((.75, 1.5), (1, 1), (.5, .5)) # Beta Binomial parameters
αs = (1.887, 1.0, 3.0) # Dirichlet Process parameters
# plt row, column
plt11 = empty_plot(k, legend = false, yaxis = yaxis_scale, xrotation = xrotation, xfontsize = xfontsize, palette = colorpalet);
model_pmf!(plt11, Duniform, fill(1, k))
plot!(plt11, title = "Uniform", ylab = "Probabilty", xlab = "Model", ylims = ylimsModel);

# plt12 = empty_plot(k, legend = false, yaxis = yaxis_scale, xrotation = xrotation, xfontsize = xfontsize, palette = colorpalet);
# model_pmf!(plt12, DBetaBinom);
# plot!(plt12, title = "Beta-binomial (α = $(DBetaBinom.α), β = $(DBetaBinom.β))", xlab = "Model", ylims = ylimsModel);

plt12 = empty_plot(k, legend = :top, yaxis = yaxis_scale, xrotation = xrotation, xfontsize = xfontsize, palette = colorpalet);

for i in eachindex(αβ)
	α, β = αβ[i]
	labels = fill("", k)
	labels[1] = "α = $α, β = $β"
	model_pmf!(plt12, BetaBinomialConditionalUrnDistribution(ones(Int, k), 1, α, β), fill(i, k), labels);
end
plot!(plt12, title = "Beta-binomial", xlab = "Model", ylims = ylimsModel);
plot!(plt12, yformatter=_->"");

# plt12 = empty_plot(k, legend = false, yaxis = yaxis_scale, xrotation = xrotation, xfontsize = xfontsize, palette = colorpalet);
# model_pmf!(plt12, DBetaBinom);
# plot!(plt12, title = "Beta-binomial (α = $(DBetaBinom.α), β = $(DBetaBinom.β))", xlab = "Model", ylims = ylimsModel)

plt13 = empty_plot(k, legend = :top, yaxis = yaxis_scale, xrotation = xrotation, xfontsize = xfontsize, palette = colorpalet);
for i in eachindex(αs)
	α = αs[i]
	labels = fill("", k)
	labels[1] = "α = $α"
	DP = RandomProcessDistribution(Turing.RandomMeasures.DirichletProcess(α), k)
	model_pmf!(plt13, DP, fill(i, k), labels);
end
plot!(plt13, title = "Dirichlet Process", xlab = "Model", ylims = ylimsModel);
plot!(plt13, yformatter=_->"");

# plt13 = empty_plot(k, legend = false, yaxis = yaxis_scale, xrotation = xrotation, xfontsize = xfontsize, palette = colorpalet);
# model_pmf!(plt13, DDirichletProcess);
# plot!(plt13, title = "DirichletProcess (α = $(DDirichletProcess.rpm.α))", xlab = "Model", ylims = ylimsModel);

function setAxis!(plt, addLabels = false)
	if addLabels
		plot!(plt, xlab = "No. inequalities", ylab = "Probabilty", ylims = ylimsIncl);
	else
		plot!(plt, ylims = ylimsIncl, yformatter = _->"");
	end
end

plt21 = plot();
incl_pmf!(plt21, Duniform, yaxis = yaxis_scale, palette = colorpalet, color = fill(1, k));
setAxis!(plt21, true);

plt22 = plot();
for i in eachindex(αβ)
	α, β = αβ[i]
	labels = fill("", k)
	labels[1] = "α = $α, β = $β"
	incl_pmf!(plt22, BetaBinomialConditionalUrnDistribution(ones(Int, k), 1, α, β), yaxis = yaxis_scale, palette = colorpalet, color = fill(i, k))
end
setAxis!(plt22);

plt23 = plot();
for i in eachindex(αβ)
	α = αs[i]
	labels = fill("", k)
	labels[1] = "α = $α"
	DP = RandomProcessDistribution(Turing.RandomMeasures.DirichletProcess(α), k)
	incl_pmf!(plt23, DP, yaxis = yaxis_scale, palette = colorpalet, color = fill(i, k))
end
setAxis!(plt23);


# plt23 = incl_pmf(DDirichletProcess, yaxis = yaxis_scale, palette = colorpalet);
# plot!(plt23, xlab = "No. inequalities", ylims = ylimsIncl);

w = 420
joint = plot(plt11, plt12, plt13, plt21, plt22, plt23, layout = (2, 3), size = (2w, 3w))
savefig(joint, joinpath("figures", "prior_$k.pdf"))


# ys = Vector[rand(10), rand(20)]# .* u"km"
# plot(ys, color=[:black :orange], line=(:dot, 4), marker=([:hex :d], 12, 0.8, Plots.stroke(3, :gray)))

# rework of the code above using only multivariate distributions
using EqualitySampler, Distributions, Plots
import DataFrames as DF
import Turing, CSV

updateDistribution(d::UniformMvUrnDistribution, args) = d
updateDistribution(d::BetaBinomialMvUrnDistribution, args) = BetaBinomialMvUrnDistribution(d.k, args...)
updateDistribution(d::RandomProcessMvUrnDistribution, args) = RandomProcessMvUrnDistribution(d.k, Turing.RandomMeasures.DirichletProcess(args...))

make_title(::UniformMvUrnDistribution) = "Uniform"
make_title(d::BetaBinomialMvUrnDistribution) = "BetaBinomial α=$(d.α) β=$(d.β)"
make_title(d::RandomProcessMvUrnDistribution) = "Dirichlet Process α=$(d.rpm.α)"

k=5
dists = (
	(
		dist = UniformMvUrnDistribution(k),
		args = ((k,),)
	),
	(
		dist = BetaBinomialMvUrnDistribution(k),
		args = ((.75, 1.5), (1, 1), (.5, .5), (1, 5))
	),
	(
		dist = RandomProcessMvUrnDistribution(k, Turing.RandomMeasures.DirichletProcess(1.887)),
		args = ((0.5,), (1.887,), (1.0,), (3.0,))
	)
)


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
			model_probs = vec(mapslices(m->logpdf(D, m), models, dims = 1))
			incl_probs = logpdf_incl.(Ref(D), 0:k-1)

			nm = make_title(D)
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

df_wide_model_probs, df_long_model_probs, df_wide_incl_probs, df_long_incl_probs = get_data(dists)

CSV.write(joinpath("tables", "model_probs_figure_2.csv"), df_wide_model_probs)
CSV.write(joinpath("tables", "incl_probs_figure_2.csv"),  df_wide_incl_probs)

# TODO actually remake figure 2
