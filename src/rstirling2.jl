#=

	r-Stirling numbers of the second kind

=#

stirlings2rExplLogTerm(n::T, k::T, r::T, j::T)   where T <: Integer = logbinomial(n - r, j) + logstirlings2(j, k - r) + log(r) * (n - r - j)
stirlings2rExplLogTermr0(n::T, k::T, r::T, j::T) where T <: Integer = logbinomial(n - r, j) + logstirlings2(j, k - r)
stirlings2rExplTerm(n::T, k::T, r::T, j::T)      where T <: Integer =    binomial(n - r, j) *    stirlings2(j, k - r) *     r  ^ (n - r - j)

# just for precomputing
_stirlings2rExplTerm_precompute(n::T, k::T, r::T, j::T)      where T <: Integer =    binomial(n - r, j) *    _stirlings2_inner(j, k - r) *     r  ^ (n - r - j)


"""
	stirlings2r(n::T, k::T, r::T) where T <: Integer
	stirlings2r(n::T, k::T, r::T, ::Type{ExplicitStrategy})  where T <: Integer
	stirlings2r(n::T, k::T, r::T, ::Type{RecursiveStrategy}) where T <: Integer

Compute the r-Stirlings numbers of the second kind.
The `ExplicitStrategy` (default) uses an explicit loop and is computationally more efficient but subject to overflow, so using BigInt is advised.
The `RecursiveStrategy` uses recursion and is mathematically elegant yet inefficient for large values.
"""
stirlings2r(n::T, k::T, r::T) where T <: Integer = stirlings2r(n, k, r, ExplicitStrategy)
function stirlings2r(n::T, k::T, r::T, ::Type{ExplicitStrategy}) where T <: Integer

	succes, value = stirlings2r_base_cases(n, k, r)
	succes && return value

	return sum(j->stirlings2rExplTerm(n, k, r, j), one(T):n-r)

end

_stirlings2r_precompute(n::T, k::T, r::T) where T <: Integer = sum(j->_stirlings2rExplTerm_precompute(n, k, r, j), 1:n-r)

stirlings2r(n::T, k::T, r::T, ::Type{RecursiveStrategy}) where T <: Integer = stirlings2r_recursive(n, k, r)

# required for memoization
function stirlings2r_recursive(n::T, k::T, r::T) where T <: Integer

	succes, value = stirlings2r_base_cases(n, k, r)
	succes && return value
	return k * stirlings2r_recursive(n - 1, k, r) + stirlings2r_recursive(n - 1, k - 1, r)

end

"""
Base cases for stirlings2r. Should return a tuple of (base_case_found::Bool, value::T).
"""
function stirlings2r_base_cases(n::T, k::T, r::T) where T <: Integer

	n < r							&& return (true, zero(T))
	(k > n || k < r)				&& return (true, zero(T))
	n == k							&& return (true, one(T))
	iszero(n) || iszero(k)			&& return (true, zero(T))
	n == r							&& return (true, one(T))
	k == r							&& return (true, r^(n - r))
	k == r + 1						&& return (true, (r + 1)^(n - r) - r^(n - r)) # <- fix me!
	iszero(r) 						&& return (true, stirlings2(n, k))

	if n <= _r_stirling2r_N_MAX && r <= _r_stirling2r_R_MAX
		index = _stirling2r_index(n, k, r)
		if 1 <= index <= length(_stirlings2r_table_BigInt)
			return (true, T(_stirlings2r_table_BigInt[index]))
		end
	end

	return (false, zero(T))
end


"""
	logstirlings2r(n::T, k::T, r::T) where T <: Integer

Computes the logarithm of the r-stirling numbers with an explicit formula.
"""
function logstirlings2r(n::T, k::T, r::T) where T <: Integer

	iszero(r) && return logstirlings2(n, k)

	succes, value = stirlings2r_base_cases(n, k, r)
	succes && value >= zero(T) && return log(value)

	return logsumexp_batch(map(j->stirlings2rExplLogTerm(n, k, r, j), one(T):n-r))

end

# TODO: did the definition get deleted?
# """
# 	stirlings2r_recursive(n::T, k::T, r::T) where T <: Integer

# Computes the r-stirling numbers with a recursive method. Uses memoization for efficiency.
# """
#endregion
