using EqualitySampler, Plots, GraphRecipes, Colors, LazySets, Measures
import StatsBase: countmap
import OrderedCollections

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

	A = Matrix{T}(undef, k, k)
	for i in 1:k-1
		A[i, i] = 1
		for j in i+1:k
			A[i, j] = A[j, i] = model[i] == model[j] ? 1 : 0
		end
	end
	A[k ,k] = 1
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

get_colors(k::Int) = distinguishable_colors(k, [RGB(128/255,128/255,128/255)])

function plot_model(model::AbstractVector{T}) where T<:Integer

	k = length(model)
	A = zeros(T, k, k)
	for i in 1:k
		A[i, i] = 1
	end

	# TODO: this should take the model as an argument!
	colors = get_colors(k)

	x, y = get_xy(k)
	plt = plot(background_color_inside = plot_color(:lightgrey, 0.15), margin = 0.01mm)
	# TODO: not sure wheter all this OrderedDict is necessary
	tb = sort(OrderedCollections.OrderedDict(countmap(model)), byvalue=true, lt = !isless)
	count = 1
	for (k, v) in tb
		if v > 1
			shape = make_shape(model, k, x, y)
			plot!(plt, shape, alpha = .5, fillcolor = colors[count], linealpha = 0.0)
			count += 1
		end
	end
	graphplot!(plt, A, x = x, y = y, markersize = 0.2, nodeshape=:circle, fontsize = 10, linecolor = :darkgrey, nodecolor = plot_color(:white, 0.0), curves = false)
	return plt
end

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

	hull = convex_hull(points)
	return VPolygon(hull)
end


function make_grid(k; max_per_row = 5)

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

	g = Plots.GridLayout(total_no_rows, 1)
	for i in 1:total_no_rows
		g[i, 1] = Plots.GridLayout(1, no_cols[i])
	end

	max_no_cols = maximum(no_cols)
	return g, total_no_rows, max_no_cols

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

function plot_modelspace(k::Integer, save::Bool = true)

	models = generate_distinct_models(k)
	order = sortperm(ordering.(eachcol(models)), lt = !isless)
	plots = [plot_model(view(models, :, i)) for i in order]
	layout, max_rows, max_cols = make_grid(k)
	w = 100
	plt = plot(plots..., layout = layout, size = (max_cols*w, max_rows*w))
	save && savefig(plt, "figures/modelspace_$k.pdf")
	return plt
end

plot_modelspace(5, false)

for k in 3:6
	plot_modelspace(k)
end
