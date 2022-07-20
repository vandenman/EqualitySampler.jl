"""
$(TYPEDSIGNATURES)

Compute the absolute value of the Stirlings numbers of the first kind.
The `EqualitySampler.ExplicitStrategy` (default) uses an explicit loop and is computationally more efficient but subject to overflow, so using BigInt is advised.
The `EqualitySampler.RecursiveStrategy` uses recursion and is mathematically elegant yet inefficient for large values.
"""
unsignedstirlings1(n::T, k::T) where T <: Integer = unsignedstirlings1(n, k, ExplicitStrategy)
unsignedstirlings1(n::T, k::U) where {T<:Integer, U<:Integer} = unsignedstirlings1(promote(n, k)...)

"""
$(TYPEDSIGNATURES)

Compute the logarithm of the absolute value of the Stirlings numbers of the first kind.
"""
logunsignedstirlings1(n::T, k::U) where {T<:Integer, U<:Integer} = logunsignedstirlings1(promote(n, k)...)

function unsignedstirlings1_base_cases(n::T, k::T) where T <: Integer

	# adapted from Combinatorics.stirlings1
	n < 0 || k > n						&& return (true, zero(T))
	n == k == zero(T)					&& return (true, one(T))
	n == zero(T) || k == zero(T)		&& return (true, zero(T))
	n == k								&& return (true, one(T))
	k == one(T)							&& return (true, T(factorial(n - 1)))
	k == 2								&& return (true, round(T, factorial(n - 1) * sum(i->1/i, 1:n-1)))
	k == n - 1							&& return (true, T(binomial(n, 2)))
	k == n - 2							&& return (true, T(div((3 * n - 1) * binomial(n, 3), 4)))
	k == n - 3							&& return (true, T(binomial(n, 2) * binomial(n, 4)))

	if 7 <= n <= _stirling1_N_MAX && 3 <= k <= n - 3
		index = _stirling1_index(n, k)
		return (true, T(_stirlings1_table_BigInt[index]))
	end

	return(false, zero(T))
end

unsignedstirlings1(n::T, k::T, ::Type{RecursiveStrategy}) where T <: Integer = unsignedstirlings1_recursive(n, k)

function unsignedstirlings1_recursive(n::T, k::T) where T <: Integer

	succes, value = unsignedstirlings1_base_cases(n, k)
	succes && return value
	return (n - 1) * unsignedstirlings1_recursive(n - 1, k) + unsignedstirlings1_recursive(n - 1, k - 1)

end

function unsignedstirlings1(n::T, k::T, ::Type{ExplicitStrategy}) where T <: Integer

	# 8.20 of  Charalambides, C. A. (2018). Enumerative combinatorics. CRC Press.

	succes, value = unsignedstirlings1_base_cases(n, k)
	succes && return value

	return _unsignedstirlings1_precompute(n, k)

end

function _unsignedstirlings1_precompute(n::T, k::T) where T <: Integer
	sign = (-1)^(n - k)
	result = zero(T)
	for r in 0:n-k
		result += sign * stirlings1ExplTerm(n, k, r)
		sign *= -1
	end
	return result
end


stirlings1ExplTerm(n::T, k::T, r::T)    where T <: Integer =    binomial(n + r - 1, k - 1) *    binomial(2n - k, n - k - r) *    stirlings2(n - k + r, r)
stirlings1ExplLogTerm(n::T, k::T, r::T) where T <: Integer = logbinomial(n + r - 1, k - 1) + logbinomial(2n - k, n - k - r) + logstirlings2(n - k + r, r)

function logunsignedstirlings1(n::T, k::T) where T <: Integer

	# avoids an explicit factorial in unsignedstirlings1_base_cases
	n != zero(T) && k == 1		&& return SpecialFunctions.logfactorial(n - 1)
	n >= 2 && k == 2			&& return SpecialFunctions.logfactorial(n - 1) + log(sum(i->1/i, 1:n-1))

	succes, value = unsignedstirlings1_base_cases(n, k)
	succes && value >= zero(T) && return log(value)

	# this is more accurate than the stirlings1ExplLogTerm which introduces floating point errors for some reason
	res = log(unsignedstirlings1(BigInt(n), BigInt(k)))
	if T === BigInt
		return res
	end
	return Float64(res)
	# terms = map(r->stirlings1ExplLogTerm(n, k, r), 0:n-k)
	# return alternating_logsumexp_batch(terms)
end
