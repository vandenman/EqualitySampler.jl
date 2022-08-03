"""
	count_combinations(k::Int, islands::Int)
	count_combinations(s::AbstractString)
	count_combinations(x::AbstractVector)

Count the number of duplicate configurations that specify an identical model.
"k" specifes the number of variables and "islands" specifies the number of unequal clusters of variables.
For example, given "1122" we have k = 4 and islands = 2.
Note that s may contains spaces, e.g., "11 22 33" will be interpreted as "112233"

# Examples
```julia
julia> count_combinations(3, 1)
3
julia> count_combinations("111") # 111, 222, 333
3
```
"""
count_combinations(k::T, islands::T) where T<:Integer = factorial(islands) * binomial(k, islands)
count_combinations(k::T, islands::U) where {T<:Integer, U<:Integer} = count_combinations(promote(k, islands)...)
log_count_combinations(k::T, islands::T) where T<:Integer = SpecialFunctions.logfactorial(islands) + logbinomial(k, islands)
log_count_combinations(k::T, islands::U) where {T<:Integer, U<:Integer} = log_count_combinations(promote(k, islands)...)

function count_combinations(s::AbstractString)
	s = filter(!isspace, s)
	k = length(s)
	islands = no_distinct_groups_in_partition(s)
	return count_combinations(k, islands)
end
count_combinations(x::AbstractVector) = count_combinations(length(x), no_distinct_groups_in_partition(x))

log_count_combinations(x::AbstractVector) = log_count_combinations(length(x), no_distinct_groups_in_partition(x))



# abstract type PartitionCountingSorting end
# struct NoSorting <: PartitionCountingSorting end
# struct DefaultSorting <: PartitionCountingSorting end

"""
count_set_partitions_given_partition_size(n::T) where T<:Integer

Given a maximum partition size n, counts the number of partitions that have length 1:n.
count_set_partitions_given_partition_size(n) is more efficient than count_set_partitions_given_partition_size.(n, 1:n).
The implementation is based on the mathematica code in https://oeis.org/A036040
"""
function count_set_partitions_given_partition_size(f!::Function, n::T, sorted::Bool = true) where T<:Integer
	part = Combinatorics.integer_partitions(n)
	f!(part)
	if sorted
		sort!.(part; rev=true)
		reverse!(part)
		sort!(part; by=length)
	end

	# @show part

	result = Vector{T}(undef, length(part))
	for i in eachindex(result)
		result[i] = Combinatorics.multinomial(T.(part[i])...) ÷ mapreduce(x->factorial(T(x)), *, runs(part[i]))
	end
	return result, part
end
runs(x) = values(StatsBase.StatsBase.countmap(x))
count_set_partitions_given_partition_size(n::T, sorted::Bool = true) where T = count_set_partitions_given_partition_size(identity, n, sorted)
