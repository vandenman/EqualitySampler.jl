"""
	generate_distinct_models(k::Int)

Generates all distinct models that represent equalities.
"""
function generate_distinct_models(k::Int)
	# based on https://stackoverflow.com/a/30898130/4917834
	# TODO: return a generator rather than directly all results
	current = ones(Int, k)
	no_models = bellnum(k)
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

abstract type AbstractPartitionSpace end
struct DistinctPartitionSpace <: AbstractPartitionSpace end
struct DuplicatedPartitionSpace <: AbstractPartitionSpace end

struct PartitionIterator{T<:Integer, P<:AbstractPartitionSpace}
	no_models::T
	current_model::Vector{T}
	function PartitionIterator(k::T, P::Type{U}) where {T<:Integer, U<:AbstractPartitionSpace}
		new{T, U}(bellnum(k), ones(T, k))
	end
end

partition_space(k::Integer; distinct::Bool = true) = PartitionIterator(k, distinct ? DistinctPartitionSpace : DuplicatedPartitionSpace)

# TODO: shouldn't the iterator be stateless?
function Base.iterate(iter::PartitionIterator{T, DistinctPartitionSpace}, state=1) where T<:Integer
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


Base.length(iter::PartitionIterator) = iter.no_models
Base.eltype(::Type{PartitionIterator{T, P}}) where {T, P} = Vector{T}
Base.IteratorSize(::Type{PartitionIterator{T, P}}) where {T, P} = Base.HasLength()
function Base.Matrix(iter::PartitionIterator{T, P}) where {T, P}
	res = Matrix{T}(undef, length(iter.current_model), iter.no_models)
	for (i, m) in enumerate(iter)
		res[:, i] .= m
	end
	return res
end

"""
	generate_distinct_models(k::Int)

Returns an iterator that generates all models that represent equalities, including duplicates that represent the same unique model (e.g., [1, 1, 1] and [2, 2, 2]) .
"""
function generate_all_models(k::Int)
	return Iterators.product(fill(1:k, k)...)
end
