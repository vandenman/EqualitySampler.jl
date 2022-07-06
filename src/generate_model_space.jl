"""
	generate_distinct_models(k::Int)

Generates all distinct models that represent equalities.
"""
function generate_distinct_models(k::Int)
	# based on https://stackoverflow.com/a/30898130/4917834
	# TODO: return a generator rather than directly all results
	current = ones(Int, k)
	no_models = count_distinct_models(k)
	result = Matrix{Int}(undef, k, no_models)
	result[:, 1] .= current
	isone(k) && return result
	range = k:-1:2
	for i in 2:no_models

		idx = findfirst(i->current[i] < k && any(==(current[i]), current[1:i-1]), range)
		rightmost_incrementable = range[idx]
		current[rightmost_incrementable] += 1
		current[rightmost_incrementable + 1 : end] .= 1
		result[:, i] .= current

	end
	return result
end

# TODO: document this, export it, and use it everywhere internally and externally
struct DistinctModelsIterator{T<:Integer}
	no_models::T
	current_model::Vector{T}
	function DistinctModelsIterator(k::T) where T<:Integer
		new{T}(count_distinct_models(k), ones(T, k))
	end
end

function Base.iterate(iter::DistinctModelsIterator{T}, state=1) where T<:Integer
	state > iter.no_models && return nothing
	isone(state) && return (copy(iter.current_model), state + 1)
	state == iter.no_models && return (collect(eachindex(iter.current_model)), state + 1)

	k = T(length(iter.current_model))
	current = iter.current_model
	range = k:-1:2
	i = state

	idx = findfirst(i->current[i] < k && any(==(current[i]), view(current, 1:i-1)), range)
	rightmost_incrementable = range[idx]
	current[rightmost_incrementable] += 1
	current[rightmost_incrementable + 1 : end] .= 1
	iter.current_model .= current
	return (copy(current), state + 1)
end


Base.length(iter::DistinctModelsIterator) = iter.no_models
Base.eltype(::Type{DistinctModelsIterator{T}}) where T<:Integer = Vector{T}
Base.IteratorSize(::Type{DistinctModelsIterator{T}}) where T<:Integer = Base.HasLength()
function Base.Matrix(iter::DistinctModelsIterator{T}) where T<:Integer
	res = Matrix{T}(undef, length(iter.current_model), iter.no_models)
	for (i, m) in enumerate(iter)
		res[:, i] .= m
	end
	return res
end
# doesn't work as intended, will make a Matrix{Vector{Int}}
# Base.size(iter::DistinctModelsIterator{T}) where T<:Integer = (length(iter.current_model), iter.no_models)
# Base.IteratorSize(::Type{DistinctModelsIterator{T}}) where T<:Integer = Base.HasShape{2}()


"""
	generate_distinct_models(k::Int)

Returns an iterator that generates all models that represent equalities, including duplicates that represent the same unique model (e.g., [1, 1, 1] and [2, 2, 2]) .
"""
function generate_all_models(k::Int)
	return Iterators.product(fill(1:k, k)...)
end
