#=

	Stirling numbers of the second kind

=#

# these two compute one term of the explicit algorithm of the stirlings2. Note that they are incorrect for j = 0!
stirlings2ExplLogTerm(n::T, k::T, j::T) where T<:Integer = logbinomial(k, j) + n * log(j)
stirlings2ExplTerm(n::T, k::T, j::T) where T<:Integer = binomial(k, j) * j^n

"""
$(TYPEDSIGNATURES)

Compute the Stirlings numbers of the second kind.
The `EqualitySampler.ExplicitStrategy` (default) uses an explicit loop and is computationally more efficient but subject to overflow, so using BigInt is advised.
The `EqualitySampler.RecursiveStrategy` uses recursion and is mathematically elegant yet inefficient for large values.
"""
stirlings2(n::T, k::U, S::Type{<:StirlingStrategy} = ExplicitStrategy) where {T <: Integer, U <: Integer} = stirlings2(promote(n, k)..., ExplicitStrategy)
function stirlings2(n::T, k::T, S::Type{<:StirlingStrategy} = ExplicitStrategy) where T <: Integer

	succes, value = stirlings2_base_cases(n, k)
	succes && return value

	return _stirlings2_inner(n, k, S)
end

function _stirlings2_inner(n::T, k::T, ::Type{ExplicitStrategy}) where T <: Integer
	res = zero(T)
	sign = iseven(k) ? -1 : 1
	for j in 1:k
		term = stirlings2ExplTerm(n, k, j)
		res += sign * term
		sign *= -1
	end
	return res ÷ factorial(k)
end

function _stirlings2_inner(n::T, k::T, ::Type{RecursiveStrategy}) where T <: Integer

    #=
        TODO:
            - [ ] can reduce the number of iterations here!
            - [ ] can use stirlings2_base_cases?
    =#

    vals = zeros(T, k + one(T))
    vals[1] = one(T)
    vals[2] = one(T)
    for ni in T(3):n

        for ki in min(k + 1, ni - 1):-1:2#max(2, min(k + 1, ni - 1) - (ni - n))#max(2, ni - k - 2)
            vals[ki] = ki * vals[ki] + vals[ki - 1]
        end
        vals[min(k+1, ni-1)] = one(T)
        # @show vals,  min(k + 1, ni - 1):-1:max(2, ni - k - 1)
    end

    return k * vals[k] + vals[k - 1]

end

# function stirlings2_recursive(n::T, k::T) where T <: Integer

# 	succes, value = stirlings2_base_cases(n, k)
# 	succes && return value

# 	return k * stirlings2_recursive(n - 1, k) + stirlings2_recursive(n - 1, k - 1)
# end


# function stirlings2_recursive(n::T, k::T) where T <: Integer

#     succes, value = stirlings2_base_cases(n, k)
# 	succes && return value

#     vals = zeros(T, k + 1)
#     vals[1] = one(T)
#     vals[2] = one(T)
#     for ni in 3:n
#         for ki in min(k+1, ni - 1):-1:max(2, ni - k)
#             vals[ki] = ki * vals[ki] + vals[ki - 1]
#         end
#         vals[min(k+1, ni-1)] = one(T)
#     end

#     return k * vals[k] + vals[k - 1]
# end



"""
Base cases for stirlings2. Returns a tuple of (base_case_found::Bool, value::T).
"""
function stirlings2_base_cases(n::T, k::T) where T <: Integer

	n < zero(T) || k < zero(T) 			&&		return (true, zero(T))
	k > n								&&		return (true, zero(T))
	n == k								&&		return (true, one(T))
	iszero(n) || iszero(k)				&&		return (true, zero(T))

    k == 1                              &&      return (true, one(T))
	k == 2								&&		return (true, T(2^(n-1) - 1))
	k == 3								&&		return (true, T((3^n - 3 * 2^n + 3) ÷ 6))
	k == n - 1							&&		return (true, T(binomial(n, T(2))))
    k == n - 2                          &&      return (true, T((3n - 5) * binomial(n, 3) ÷ 4))

	return (false, zero(T))

end

function stirlings2_log_base_cases(n::T, k::T) where T <: Integer

    U = float(T)

	n < zero(T) || k < zero(T) 			&&		return (true, U(-Inf))
	k > n								&&		return (true, U(-Inf))
    k == n							    &&		return (true, zero(U))
	iszero(n) || iszero(k)				&&		return (true, U(-Inf))

    k == 1                              &&      return (true, zero(U))
	k == 2								&&		return (true, U(LogExpFunctions.logsubexp(log(2) * (n - 1), 0)))
	k == 3								&&		return (
        true,
        n < 3 ? U(log((3^n - 3 * 2^n + 3) ÷ 6)) : U(
                LogExpFunctions.logaddexp(LogExpFunctions.logsubexp(n * log(3), log(3) + n * log(2)), log(3)) - log(6)
        )
    )

    k == n - 1							&&		return (true, U(logbinomial(n, T(2))))
    k == n - 2                          &&      return (true, U(log(3n - 5) + logbinomial(n, 3) - log(4)))


	return (false, zero(U))

end

"""
$(TYPEDSIGNATURES)

Compute the logarithm of the Stirlings numbers of the second kind.
"""
logstirlings2(n::T, k::U, S::Type{<:StirlingStrategy} = RecursiveStrategy) where {T<:Integer, U<:Integer} = logstirlings2(promote(n, k)..., S)
# logstirlings2(n::T, k::T) where T <: Integer = logstirlings2(n, k, RecursiveStrategy)

function logstirlings2(n::T, k::T, S::Type{<:StirlingStrategy} = RecursiveStrategy) where {T <: Integer}

	succes, value = stirlings2_log_base_cases(n, k)
	succes && return value

    return logstirlings2_inner(n, k, S)
	# logvalues = map(j->stirlings2ExplLogTerm(n, k, j), one(T):k)
	# return alternating_logsumexp_batch(logvalues) - SpecialFunctions.logfactorial(k)

end

function logstirlings2_inner(n::T, k::T, ::Type{ExplicitStrategy}) where T <: Integer

	logvalues = map(j->stirlings2ExplLogTerm(n, k, j), one(T):k)
	return alternating_logsumexp_batch(logvalues) - SpecialFunctions.logfactorial(k)

end

function logstirlings2_inner(n::T, k::T, ::Type{RecursiveStrategy}) where T <: Integer

    U = float(T)
    vals = zeros(U, k + 1)
    vals[1] = zero(U)
    vals[2] = zero(U)
    for ni in 3:n
        for ki in min(k+1, ni - 1):-1:2#max(2, ni - k)
            vals[ki] = LogExpFunctions.logaddexp(log(ki) + vals[ki], vals[ki - 1])
        end
        vals[min(k+1, ni-1)] = zero(U)
        # @show vals
    end
    # @show vals[k - 3], vals[k - 2]
    return LogExpFunctions.logaddexp(log(k) + vals[k], vals[k - 1])
end
