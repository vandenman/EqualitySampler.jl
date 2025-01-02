abstract type StirlingStrategy end
struct RecursiveStrategy <: StirlingStrategy end
struct ExplicitStrategy <: StirlingStrategy end

function alternating_logsumexp_batch(x::AbstractVector{T}) where T <: Number
	isone(length(x)) && return @inbounds first(x)
	# accounts for the flipping 1, -1
	# there is also a streaming version, see http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html
	result = zero(T)
	# sign = isodd(length(x) - 1) ? -1 : 1
	sign = iseven(length(x)) ? -1 : 1
	range = sign == -1 ? (1:2:length(x)) : (2:2:length(x))
	alpha = maximum(view(x, range))
	# alpha = maximum(x)
	@inbounds for i in eachindex(x)
		result += sign * exp(x[i] - alpha)
		sign *= -1
	end
	# TODO: come up with something better than this fallback!
	if result < zero(T)
		@show x
		@warn "alternating_logsumexp_batch failed which likely introduced numerical errors."
		return T(log(sum(value->exp(BigFloat(value)), x)))
	else
		return log(result) + alpha
	end
	# return log(sum((sign, value) -> sign * exp(value - alpha), zip(x, it))) + alpha
end

