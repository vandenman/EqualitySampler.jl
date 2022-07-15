#=

	Stirling numbers of the second kind

=#

# these two compute one term of the explicit algorithm of the stirlings2. Note that they are incorrect for j = 0!
stirlings2ExplLogTerm(n::T, k::T, j::T) where T<:Integer = logbinomial(k, j) + n * log(j)
stirlings2ExplTerm(n::T, k::T, j::T) where T<:Integer = binomial(k, j) * j^n

"""
	stirlings2(n::T, k::T) where T <: Integer
	stirlings2(n::T, k::T, ::Type{EqualitySampler.ExplicitStrategy})  where T <: Integer
	stirlings2(n::T, k::T, ::Type{EqualitySampler.RecursiveStrategy}) where T <: Integer

Compute the Stirlings numbers of the second kind.
The `EqualitySampler.ExplicitStrategy` (default) uses an explicit loop and is computationally more efficient but subject to overflow, so using BigInt is advised.
The `EqualitySampler.RecursiveStrategy` uses recursion and is mathematically elegant yet inefficient for large values.
"""
stirlings2(n::T, k::T) where T <: Integer = stirlings2(n, k, ExplicitStrategy)
function stirlings2(n::T, k::T, ::Type{ExplicitStrategy}) where T <: Integer

	succes, value = stirlings2_base_cases(n, k)
	succes && return value

	return _stirlings2_inner(n, k)
end

function _stirlings2_inner(n::T, k::T) where T <: Integer
	res = zero(T)
	sign = iseven(k) ? -1 : 1
	for j in 1:k
		term = stirlings2ExplTerm(n, k, j)
		res += sign * term
		sign *= -1
	end
	return res ÷ factorial(k)
end

stirlings2(n::T, k::T, ::Type{RecursiveStrategy}) where T <: Integer = stirlings2_recursive(n, k)

function stirlings2_recursive(n::T, k::T) where T <: Integer

	succes, value = stirlings2_base_cases(n, k)
	succes && return value

	return k * stirlings2_recursive(n - 1, k) + stirlings2_recursive(n - 1, k - 1)
end

"""
Base cases for stirlings2. Returns a tuple of (base_case_found::Bool, value::T).
"""
function stirlings2_base_cases(n::T, k::T) where T <: Integer

	n < zero(T) || k < zero(T) 			&&		return (true, zero(T))
	k > n								&&		return (true, zero(T))
	n == k								&&		return (true, one(T))
	n == zero(T) || k == zero(T)		&&		return (true, zero(T))
	k == n - 1							&&		return (true, T(binomial(n, T(2))))
	k == T(2)							&&		return (true, T(2^(n-1) - 1))

	if n <= _r_stirling2r_N_MAX
		index = _stirling2r_index(n, k, zero(T))
		if 1 <= index <= length(_stirlings2r_table_BigInt)
			return (true, T(_stirlings2r_table_BigInt[index]))
		end
	end

	return (false, zero(T))

end

"""
	logstirlings2(n::T, k::T) where T <: Integer

Compute the logarithm of the Stirlings numbers of the second kind with an explicit formula.
"""
function logstirlings2(n::T, k::T) where T <: Integer

	succes, value = stirlings2_base_cases(n, k)
	succes && value >= zero(T) && return log(value)

	logvalues = map(j->stirlings2ExplLogTerm(n, k, j), one(T):k)
	return alternating_logsumexp_batch(logvalues) - SpecialFunctions.logfactorial(k)

end
logstirlings2(n::T, k::U) where {T<:Integer, U<:Integer} = logstirlings2(promote(n, k)...)
