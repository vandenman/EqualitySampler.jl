#region helperfunctions

abstract type StirlingStrategy end
struct RecursiveStrategy <: StirlingStrategy end
struct ExplicitStrategy <: StirlingStrategy end

logbinomial(n::Integer, k::Integer) = SpecialFunctions.logabsbinomial(n, k)[1]

# for alternating sums
alternatingIterator(x) = Iterators.cycle(isodd(length(x) - 1) ? (-1.0, 1.0) : (1.0, -1.0))

function alternating_logsumexp_batch(x::AbstractVector{T}) where T <: Number
	isone(length(x)) && return x[1]
	# accounts for the flipping 1, -1
	# there is also a streaming version, see http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html
	result = zero(T)
	sign = isodd(length(x) - 1) ? -1 : 1
	range = sign == -1 ? (1:2:length(x)) : (2:2:length(x))
	alpha = maximum(view(x, range))
	# alpha = maximum(x)
	for i in eachindex(x)
		result += sign * exp(x[i] - alpha)
		sign *= -1
	end
	# TODO: come up with something better than this fallback!
	if result < zero(T)
		@warn "alternating_logsumexp_batch failed which likely introduced numerical errors."
		return T(log(sum(value->exp(BigFloat(value)), x)))
	else
		return log(result) + alpha
	end
	# return log(sum((sign, value) -> sign * exp(value - alpha), zip(x, it))) + alpha
end

# TODO: compare this to LogExpFunctions.logsumexp
# https://github.com/JuliaStats/LogExpFunctions.jl/blob/master/src/logsumexp.jl
# probably better if we don't have to reinvent the wheel
function logsumexp_batch(x::AbstractVector{T}) where T <: Number
	isone(length(x)) && return x[1]
	alpha = maximum(x)
	return log(sum(value -> exp(value - alpha), x)) + alpha
end
