abstract type AbstractPartitionSpace end
struct DistinctPartitionSpace <: AbstractPartitionSpace end
struct DuplicatedPartitionSpace <: AbstractPartitionSpace end

"""
$(TYPEDEF)
```julia
# constructor
PartitionSpace(k::T, ::Type{U} = EqualitySampler.DistinctPartitionSpace)
```

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
	return ones(T, iter.k), ones(T, iter.k)
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
	current[idx] += one(T)
	current[idx + 1 : end] .= one(T)

	return (current, copy(current))

end

function Base.iterate(iter::PartitionSpace{T, DuplicatedPartitionSpace}, states) where T

	@inbounds begin

		current = copy(states)
		i = one(T)
		while i <= iter.k && current[i] == iter.k
			i += one(T)
		end

		i > iter.k && return nothing

		current[i] += one(T)
		current[1:i-1] .= one(T)

	end

	return current, copy(current)

end

Base.length(iter::PartitionSpace{<:Any, DistinctPartitionSpace}) = bellnum(iter.k)
Base.length(iter::PartitionSpace{<:Any, DuplicatedPartitionSpace}) = iter.k^iter.k

Base.eltype(::Type{<:PartitionSpace{T}}) where T = Vector{T}
# Base.eltype(::PartitionSpace{T}) where T = Vector{T}

# default
# Base.IteratorSize(::Type{PartitionSpace}) = Base.HasLength()

function Base.Matrix(iter::PartitionSpace{T, <:Any}) where T
    # T may be to small to represent bellnum(iter.k)
    k_int = promote_type(Int, T)(iter.k)
	res = Matrix{T}(undef, iter.k, bellnum(k_int))
	for (i, m) in enumerate(iter)
		res[:, i] .= m
	end
	return res
end
