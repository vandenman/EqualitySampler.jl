abstract type AbstractPartitionSpace end
struct DistinctPartitionSpace <: AbstractPartitionSpace end
struct DuplicatedPartitionSpace <: AbstractPartitionSpace end

"""
PartitionSpace{T<:Integer, P<:EqualitySampler.AbstractPartitionSpace}

A type to represent the space of partitions.
`EqualitySampler.AbstractPartitionSpace` indicates whether partitions should contains duplicates or be distinct.
For example, the distinct iterator will return `[1, 1, 2]` but not `[2, 2, 1]` and `[1, 1, 3]`, which are returned when `P = EqualitySampler.DuplicatedPartitionSpace`.
"""
struct PartitionSpace{T<:Integer, P<:AbstractPartitionSpace}
	k::T
	function PartitionSpace(k::T, ::Type{U} = DistinctPartitionSpace) where {T<:Integer, U<:AbstractPartitionSpace}
		k < one(k) && throw(DomainError(k, "k must be larger than zero."))
		new{T, U}(k)
	end
end

function Base.iterate(iter::PartitionSpace{T, P}) where {T, P}
	return ones(Int, iter.k), ones(Int, iter.k)
end

@inbounds function manual_findfirst(current)

	# a manual version of
	# idx = findfirst(i->current[i] < k && any(==(current[i]), view(current, 1:i-1)), range)
	# range[idx]

	k = length(current)
	range = k:-1:2
	# for (idx, i) in enumerate(range)
	for i in range
		current[i] == k && continue
		for j in 1:i-1
			if current[j] == current[i]
				return i
			end
		end
	end
	return nothing
end

@inbounds function Base.iterate(::PartitionSpace{T, DistinctPartitionSpace}, state) where T<:Integer

	current = copy(state)

	idx = manual_findfirst(current)
	isnothing(idx) && return nothing
	current[idx] += 1
	current[idx + 1 : end] .= 1

	return (current, copy(current))

end

function Base.iterate(iter::PartitionSpace{T, DuplicatedPartitionSpace}, states) where T

	@inbounds begin

		current = copy(states)
		i = 1
		while i <= iter.k && current[i] == iter.k
			i += 1
		end

		i > iter.k && return nothing

		current[i] += 1
		current[1:i-1] .= 1

	end

	return current, copy(current)

end

Base.length(iter::PartitionSpace{T, DistinctPartitionSpace})   where T = bellnum(iter.k)
Base.length(iter::PartitionSpace{T, DuplicatedPartitionSpace}) where T = iter.k^iter.k

Base.eltype(::Type{PartitionSpace{T, P}}) where {T, P} = Vector{Int}
Base.IteratorSize(::Type{PartitionSpace{T, P}}) where {T, P} = Base.HasLength()

function Base.Matrix(iter::PartitionSpace{T, P}) where {T, P}
	res = Matrix{Int}(undef, iter.k, length(iter))
	for (i, m) in enumerate(iter)
		res[:, i] .= m
	end
	return res
end
