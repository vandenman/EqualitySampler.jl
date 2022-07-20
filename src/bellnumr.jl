"""
$(TYPEDSIGNATURES)

Computes the ``r``-Bell numbers.
"""
function bellnumr(n::T, r::T) where T <: Integer

	succes, value = bellnumr_base_cases(n, r)
	succes && return value

	return bellnumr_inner(n, r)
end
bellnumr(n::T, r::U) where {T <: Integer, U <: Integer} = bellnumr(promote(n, r)...)

function bellnumr_inner(n::T, r::T) where T <: Integer

	res = zero(T)
	for k in 0:n
		res += stirlings2r(n+r, k+r, r)
	end
	return res

end

"""
$(TYPEDSIGNATURES)

Computes the logarithm of the ``r``-Bell numbers.
"""
function logbellnumr(n::T, r::T) where T <: Integer

	succes, value = bellnumr_base_cases(n, r)
	succes && return Float64(log(value))

	values = Vector{Float64}(undef, n + 1)
	for k in 0:n
		values[k+1] = logstirlings2r(n+r, k+r, r)
	end
	return LogExpFunctions.logsumexp(values)
end
logbellnumr(n::T, r::U) where {T <: Integer, U <: Integer} = logbellnumr(promote(n, r)...)

function bellnumr_base_cases(n::T, r::T) where T <: Integer

	# base cases
	n == 0		&& 		return (true, 		one(T))
	n == 1		&&		return (true, 		T(r + 1))
	# https://oeis.org/A002522 offset by 1
	n == 2 		&& 		return (true, 		T((r + 1)^2 + 1))
	# https://oeis.org/A005491 offset by 1
	n == 3		&&		return (true, 		T((r + 1)^3 + 3(r + 1) + 1))
	# https://oeis.org/A005492 simplify equations for a(n)
	n == 4		&&		return (true, 		T(15 + r * (20 + r * (12 + r * (4 + r)))))

	if 0 <= r < size(_bellnumr_table_BigInt, 1) && 5 <= n < size(_bellnumr_table_BigInt, 2)
		return (true, T(_bellnumr_table_BigInt[r+1, n-4]))
	end

	return (false, zero(T))
end

"""
$(TYPEDSIGNATURES)

Computes the Bell numbers.
"""
bellnum(n::T) where {T <: Integer} = bellnumr(n, zero(T))
