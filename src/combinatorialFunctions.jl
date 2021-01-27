#region Bell numbers and Stirling numbers
"""
	stirlings2(n::T, k::T) where T <: Integer
	
	Identical to Combinatorics.stirlings2, but uses memoization for efficiency and allows for custom integer types for precision
"""
Memoize.@memoize function stirlings2(n::T, k::T) where T <: Integer 

	if n < zero(T)
		throw(DomainError(n, "n must be nonnegative"))
	elseif n == k == zero(T)
		return one(T)
	elseif n == zero(T) || k == zero(T)
		return zero(T)
	elseif k == n - 1
		return binomial(n, T(2))
	elseif k == T(2)
		return 2^(n-1) - 1
	end

	return k * stirlings2(n - 1, k) + stirlings2(n - 1, k - 1)
end

"""
	stirling2r(n::T, k::T, r::T) where T <: Integer
	
	Computes the r-stirling numbers. Uses memoization for efficiency.
"""
Memoize.@memoize function stirlings2r(n::T, k::T, r::T) where T <: Integer

	n < r && return zero(T)
	(k > n || k < r) && return zero(T)
	n == k && return one(T)
	iszero(n) || iszero(k) && return zero(T)
	n == r && return one(T)
	k == r && return r^(n - r)
	return k * stirlings2r(n - 1, k, r) + stirlings2r(n - 1, k - 1, r) 

end

"""
	bellnumr(n::T, r::T) where T <: Integer

	Computes the r-Bell numbers.
"""
Memoize.@memoize function bellnumr(n::T, r::T) where T <: Integer
	#= 
		TODO: this could also use stirlings2r(n, k, 1), or just sum over them!

	=#
	res = zero(T)
	for k in 0:n, i in 0:n
		res +=
			binomial(n, i) *
			stirlings2(i, k) *
			r^(n - i)
	end
	return res
end
#endregion

#region combinatorial functions over the model space
"""
	count_combinations(k::Int, islands::Int)
	count_combinations(s::AbstractString)

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
count_combinations(k::Int, islands::Int) = factorial(islands) * binomial(k, islands)
function count_combinations(s::AbstractString)
	s = filter(!isspace, s)
	k = length(s)
	islands = length(unique(s))
	return count_combinations(k, islands)
end
count_combinations(x::AbstractVector) = count_combinations(length(x), length(unique(x)))

"""
	count_distinct_models(k::Int)

returns the n-th bell number, which representats the total number of unique models.
"""
count_distinct_models(k::Int) = bellnumr(k, 0)
count_models_with_incl(k, no_equalities) = stirlings2(k, k - no_equalities) .* count_combinations.(k, k - no_equalities)
