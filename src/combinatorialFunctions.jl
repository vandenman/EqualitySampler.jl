#=

	combinatorial functions over the model space

=#
"""
	count_combinations(k::Int, islands::Int)
	count_combinations(s::AbstractString)
	count_combinations(x::AbstractVector)

Count the number of duplicate configurations that specify an identical model.
"k" specifes the number of variables and "islands" specifies the number of unequal clusters of variables.
For example, given "1122" we have k = 4 and islands = 2.
Note that s may contains spaces, e.g., "11 22 33" will be interpreted as "112233"

# Examples
```jldoctest
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
	islands = length(unique(s))
	return count_combinations(k, islands)
end
count_combinations(x::AbstractVector) = count_combinations(length(x), length(unique(x)))

log_count_combinations(x::AbstractVector) = log_count_combinations(length(x), length(unique(x)))

"""
	count_distinct_models(k::Int)

returns the n-th bell number, which representats the total number of unique models.
"""
count_distinct_models(k::Integer) = bellnumr(k, zero(k))

count_distinct_models_with_no_equalities(k::T, no_equalities::T)     where T<:Integer = stirlings2(k, k - no_equalities)
count_models_with_no_equalities(k::T, no_equalities::T)              where T<:Integer = count_distinct_models_with_no_equalities(k, no_equalities) * count_combinations(k, k - no_equalities)
log_count_distinct_models_with_no_equalities(k::T, no_equalities::T) where T<:Integer = logstirlings2(k, k - no_equalities)

count_distinct_models_with_no_parameters(k::T, no_parameters::T)     where T<:Integer = stirlings2(k, no_parameters)
count_models_with_no_parameters(k::T, no_parameters::T)              where T<:Integer = count_distinct_models_with_no_parameters(k, no_parameters) * count_combinations(k, no_parameters)
log_count_distinct_models_with_no_parameters(k::T, no_parameters::T) where T<:Integer = logstirlings2(k, no_parameters)



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
runs(x) = values(countmap(x))
count_set_partitions_given_partition_size(n::T, sorted::Bool = true) where T = count_set_partitions_given_partition_size(identity, n, sorted)

# runs(x) = sort!(collect(values(countmap(x))))

# """
# count_set_partitions_given_partition_size(n::T, m::T) where T<:Integer

# Given a maximum partition size n, counts the number of partitions that have length m.
# """
# function count_set_partitions_given_partition_size(n::T, m::T) where T<:Integer
# 	part = sort!(reverse!(sort!.(Combinatorics.integer_partitions(n); rev=true)); by = length)
# 	result = Combinatorics.multinomial(T.(part[m])...) ÷ mapreduce(x->factorial(T(x)), *, runs(part[m]))
# 	return result, part[m]
# end


# function count_set_partitions_given_partition_size2(n::T, target_size::T) where T<:Integer
# 	part = filter!(x->length(x) == target_size, Combinatorics.integer_partitions(n))
# 	result = Vector{T}(undef, length(part))
# 	for i in eachindex(result)
# 		result[i] = Combinatorics.multinomial(T.(part[i])...) ÷ mapreduce(x->factorial(T(x)), *, runs1(part[i]))
# 	end
# 	return result, part
# end
# runs1(x) = collect(values(StatsBase.countmap(x)))