#=

	some faster functions that only work for partitions
	this is mainly possible since for all partitions p of size k we have that 1 <= p[i] <= k

=#

# in a direct benchmark this function is slower than the 3 functions below, but during profiling of the mcmc sampler it is faster...
# no_distinct_groups_in_partition(partition::AbstractVector{<:Integer}) = length(Set(partition))

function no_distinct_groups_in_partition_fallback(partition::AbstractVector{<:Integer})
	# identical to length(Set(partition)) or length(unique(partition)), works for all lengths
	hash = trues(length(partition))
	no_distinct = 0
	@inbounds for p in partition
		if hash[p]
			hash[p] = false
			no_distinct += 1
		end
	end
	return no_distinct
end

const _powers_2 = 2 .^(0:62)
function no_distinct_groups_in_partition_below63(partition::AbstractVector{T}) where {T<:Integer}
	# identical to length(Set(partition)) or length(unique(partition)), only works for length(partition) < 63
	T(count_ones(mapreduce(x->_powers_2[x], |, partition)))
end
function no_distinct_groups_in_partition(partition::AbstractVector{<:Integer})
	# identical to length(Set(partition)) or length(unique(partition)) but faster
	if length(partition) <= 63
		return no_distinct_groups_in_partition_below63(partition)
	else
		return no_distinct_groups_in_partition_fallback(partition)
	end
end



function fast_countmap_partition(partition::AbstractVector{T}) where {T<:Integer}
	# identical to collect(values(sort(StatsBase.countmap(partition)))
	return filter!(!(iszero), fast_countmap_partition_incl_zero(partition))
end

function fast_countmap_partition_incl_zero(partition::AbstractVector{T}) where {T<:Integer}
	# identical to collect(values(sort(StatsBase.countmap(partition))) but also includes zeros
	res = zero(partition)
	@inbounds for p in partition
		res[p] += one(T)
	end
	return res
end
