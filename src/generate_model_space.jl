"""
	generate_distinct_models(k::Int)

Generates all distinct models that represent equalities.
Deprecated in favor of `partition_space(k::Int)`.
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

"""
	generate_distinct_models(k::Int)

Returns an iterator that generates all models that represent equalities, including duplicates that represent the same unique model (e.g., [1, 1, 1] and [2, 2, 2]) .
Deprecated in favor of `partition_space(k::Int; distinct=false)`.
"""
function generate_all_models(k::Int)
	return Iterators.product(fill(1:k, k)...)
end

abstract type AbstractPartitionSpace end
struct DistinctPartitionSpace <: AbstractPartitionSpace end
struct DuplicatedPartitionSpace <: AbstractPartitionSpace end

"""
PartitionIterator{T<:Integer, P<:EqualitySampler.AbstractPartitionSpace}

A type to represent the space of partitions.
`EqualitySampler.AbstractPartitionSpace` indicates whether partitions should contains duplicates or be distinct.
For example, the distinct iterator will return `[1, 1, 2]` but not `[2, 2, 1]` and `[1, 1, 3]`, which are returned by the duplicated iterator.
"""
struct PartitionIterator{T<:Integer, P<:AbstractPartitionSpace}
	k::T
	function PartitionIterator(k::T, ::Type{U}) where {T<:Integer, U<:AbstractPartitionSpace}
		k < one(k) && throw(DomainError(k, "k must be larger than zero."))
		new{T, U}(k)
	end
end

"""
partition_space(k::Integer, P::Type{<:AbstractPartitionSpace} = DistinctPartitionSpace)

Returns an iterator that iterates over the space of partitions.
If `P = EqualitySampler.DistinctPartitionSpace` then the iterator will return `[1, 1, 2]` but not `[2, 2, 1]` and `[1, 1, 3]`, which would be returned if `P = EqualitySampler.DuplicatedPartitionSpace`.
"""
partition_space(k::Integer, P::Type{<:AbstractPartitionSpace} = DistinctPartitionSpace) = PartitionIterator(k, P)

function Base.iterate(iter::PartitionIterator{T, P}) where {T, P}
	return ones(Int, iter.k), ones(Int, iter.k)
end

function Base.iterate(iter::PartitionIterator{T, DistinctPartitionSpace}, state) where T<:Integer

	@inbounds begin
		k = iter.k
		range = k:-1:2
		current = copy(state)
		idx = findfirst(i->current[i] < k && any(==(@inbounds current[i]), view(current, 1:i-1)), range)
		isnothing(idx) && return nothing
		rightmost_incrementable = range[idx]
		current[rightmost_incrementable] += 1
		current[rightmost_incrementable + 1 : end] .= 1
	end

	return (current, copy(current))

end

function Base.iterate(iter::PartitionIterator{T, DuplicatedPartitionSpace}, states) where T

	@inbounds begin

		current = copy(states)
		i = 1
		while current[i] == iter.k
			i += 1
		end

		i > iter.k && return nothing

		current[i] += 1
		current[1:i-1] .= 1

	end

	return current, copy(current)

end

Base.length(iter::PartitionIterator{T, DistinctPartitionSpace})   where T = bellnum(iter.k)
Base.length(iter::PartitionIterator{T, DuplicatedPartitionSpace}) where T = iter.k^iter.k

Base.eltype(::Type{PartitionIterator{T, P}}) where {T, P} = Vector{Int}
Base.IteratorSize(::Type{PartitionIterator{T, P}}) where {T, P} = Base.HasLength()

function Base.Matrix(iter::PartitionIterator{T, P}) where {T, P}
	res = Matrix{Int}(undef, iter.k, length(iter))
	for (i, m) in enumerate(iter)
		res[:, i] .= m
	end
	return res
end
