using EqualitySampler, Plots#, LazySets#, Measures
import StatsBase: countmap
import OrderedCollections
import Colors
import LazySets

"""
	adjacency_mat_from_model(model::AbstractVector{T}) where T<:Integer

Transforms partition into adjacency matrix
"""
function adjacency_mat_from_model(model::AbstractVector{T}) where T<:Integer

	k = length(model)
	A = zeros(T, k, k)
	for i in 1:k
		A[i, i] = 1
	end
	return A

end

"""
	get_xy(k::Integer)

get x and y coordinates such that the k x-y pairs form a regular polygon.
"""
function get_xy(k::Integer)
	offset = k == 4 ? pi / 4 : 0.0
	x = Float64[sin(i * 2pi / k + offset) for i in 1:k]
	y = Float64[cos(i * 2pi / k + offset) for i in 1:k]
	return x, y
end
"""
	plot_adjacency(A, x, y)
	plot_adjacency(A)

plot adjacency matrix as a network
"""
# function plot_adjacency(A, x, y)
# 	graphplot(A, x = x, y = y, markersize = 0.2, nodeshape=:circle, fontsize = 10, linecolor = :darkgrey, curves = false)
# end
# plot_adjacency(A) = plot_adjacency(A, get_xy(size(A)[1])...)

get_adj_colors(k::Int) = Colors.distinguishable_colors(k, [RGB(128/255,128/255,128/255)])

function plot_model_data(model::AbstractVector{T}) where T<:Integer


	k = length(model)
	A = zeros(T, k, k)
	for i in 1:k
		A[i, i] = 1
	end

	# TODO: this should take the model as an argument!
	# colors = get_colors(k)

	x, y = get_xy(k)
	# plt = plot(background_color_inside = plot_color(:lightgrey, 0.15), margin = 0.01mm)
	# TODO: not sure wheter all this OrderedDict is necessary
	tb = sort(OrderedCollections.OrderedDict(countmap(model)), byvalue=true, lt = !isless)
	shapes = [make_shape(model, k, x, y) for (k, v) in tb if v > 1]
	return A, x, y, shapes

end

plot_model_data(x::Int) = plot_model_data(reverse(digits(x)))
function plot_model(model::AbstractVector{T}; kwargs...) where T<:Integer
	plt = plot(background_color_inside = plot_color(:lightgrey, 0.15), margin = 0.01Plots.PlotMeasures.mm; kwargs...)
	return plot_model!(plt, model; kwargs...)
end

function plot_model!(plt, model::AbstractVector{T}; markersize = 3, markerstroke = 3, kwargs...) where T<:Integer

	k = length(model)

	colors = get_adj_colors(k)

	x, y = get_xy(k)
	# TODO: not sure wheter all this OrderedDict is necessary
	tb = sort(OrderedCollections.OrderedDict(countmap(model)), byvalue=true, lt = !isless)
	count = 1

	# @show kwargs
	for (k, v) in tb
		if v > 1
			shape = make_shape(model, k, x, y)
			plot!(plt, shape; alpha = .5, fillcolor = colors[count], linealpha = 0.0, kwargs...)
			count += 1
		end
	end
	# scatter!(plt, x, y; marker = (markersize, 1.0, :white, Plots.stroke(markerstroke, :gray)), kwargs...)
	scatter!(plt, x, y; markersize = markersize, markercolor = :white, markerstrokewidth = markerstroke, markerstrokecolor = :gray, kwargs...)
	tmp0 = Iterators.flatten(zip(x, y)) |> extrema
	tmp1 = 1.3 * maximum(abs, tmp0)
	lims = (-tmp1, tmp1)
	# @show lims
	plot!(plt; xlim = lims, ylim = lims, kwargs...)
	# graphplot!(plt, A, x = x, y = y, markersize = 0.2, nodeshape=:circle, fontsize = 10, linecolor = :darkgrey, nodecolor = plot_color(:white, 0.0), curves = false,
				# linewidth = 0)
	return plt
end

plot_model(x::Int; kwargs...) = plot_model(reverse(digits(x)); kwargs...)
plot_model!(p, x::Int; kwargs...) = plot_model!(p, reverse(digits(x)); kwargs...)


function make_shape(model, k, x, y, no_points = 32, pointscale = .15)

	# we know all points are exterior
	idx = findall(==(k), model)
	xv = view(x, idx)
	yv = view(y, idx)
	# return Shape(xv, yv)

	# this is inefficient, but it works well without too much effort
	points = Vector{Vector{Float64}}(undef, length(xv) * no_points)
	circle_x, circle_y = get_xy(no_points)
	for i in eachindex(xv)
		for j in eachindex(circle_x, circle_y)
			points[j + (i-1) * no_points] = Float64[xv[i] + pointscale * circle_x[j], yv[i] + pointscale * circle_y[j]]
		end
	end

	hull = LazySets.convex_hull(points)
	return LazySets.VPolygon(hull)
end


function make_grid(k, vertical::Bool = true; max_per_row = 5)

	incl = reverse!(EqualitySampler.expected_inclusion_counts(k))
	total = sum(incl)

	total_no_rows = sum(ceil.(Int, incl ./ max_per_row))
	no_cols = Vector{Int}(undef, total_no_rows)
	fill!(no_cols, -1)
	currentInclIdx = 1
	currentInclValue = incl[currentInclIdx]
	for i in eachindex(no_cols)
		# @show i, no_cols, currentInclValue, currentInclIdx
		if currentInclValue < max_per_row
			no_cols[i] = currentInclValue
			currentInclIdx += 1
			if currentInclIdx <= k
				currentInclValue = incl[currentInclIdx]
			end
		else
			divby = ceil(Int, currentInclValue / max_per_row)
			v = ceil(Int, currentInclValue / divby)
			no_cols[i] = v
			currentInclValue -= v
			if iszero(currentInclValue)
				currentInclIdx += 1
				currentInclValue = incl[currentInclIdx]
			end
		end
		# @show i, no_cols, currentInclValue, currentInclIdx
	end

	if vertical
		g = Plots.GridLayout(total_no_rows, 1)
		for i in 1:total_no_rows
			g[i, 1] = Plots.GridLayout(1, no_cols[i])
		end

		max_no_cols = maximum(no_cols)
		return g, total_no_rows, max_no_cols
	else

		g = Plots.GridLayout(1, total_no_rows)
		for i in 1:total_no_rows
			g[1, i] = Plots.GridLayout(no_cols[i], 1)
		end

		max_no_cols = maximum(no_cols)
		return g, max_no_cols, total_no_rows
	end
end

function make_grid_5(k, vertical::Bool = true; max_per_row::Integer = 5)
	incl = reverse!(EqualitySampler.expected_inclusion_counts(k))
	total_no_rows = sum(ceil.(Int, incl ./ max_per_row))

	l = Matrix{NamedTuple{(:label, :blank), Tuple{Symbol, Bool}}}(undef, total_no_rows, max_per_row)
	# only works for k = 5
	for i in (1, 12), j in (1, 2, 4, 5)
		l[i, j] = (label = :_, blank = true)
	end
	for i in (1, 12), j in 3
		l[i, j] = (label = Symbol(1), blank = false)
	end

	for j in 1:5, i in 2:11
		l[i, j] = (label = :a, blank = false)
	end

	if vertical
		return l
	else
		return permutedims(l)
	end
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

function plot_modelspace(k::Integer, save::Bool = true, vertical::Bool = true)

	models = Matrix(PartitionSpace(k))
	order = sortperm(ordering.(eachcol(models)), lt = !isless)
	plots = [plot_model(view(models, :, i)) for i in order]
	layout, max_rows, max_cols = make_grid(k, vertical)
	w = 100
	plt = plot(plots..., layout = layout, size = (max_cols*w, max_rows*w))
	if save
		orientation = vertical ? "vertical" : "horizontal"
		name = "figures/modelspace_$(k)_$(orientation).pdf"
		savefig(plt, name)
	end
	return plt
end

# plot_modelspace(5, false, false)

# for k in 3:6
# 	plot_modelspace(k, true, false)
# end

function lexicographic_order(x)
	res = 0
	n = length(x)
	for i in eachindex(x)
		res += x[n - i + 1] * 10^(i-1)
	end
	return res
end


function ordering2(x)
	# TODO: ensure this is the actual ordering like on wikipedia
	d = countmap(x)
	res = Float64(length(x) - length(d))

	v = sort!(collect(values(d)), lt = !isless)
	# res = 0.0
	for i in eachindex(v)
		res += v[i] ./ 10 .^ (i)
	end
	res *= 10^length(x)
	res += lexicographic_order(x)
	return res
end
