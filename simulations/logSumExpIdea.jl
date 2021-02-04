import SpecialFunctions

#=

	TODO
		use explicit forms for all stirlings2 functions!
		also define logstirlings2!

		see if we can also find / use explicit forms for stirlings2r and logstirlings2r

		afterward translate this to bellnumber and rbellnumber

=#

logbinomial(n::Integer, k::Integer) = SpecialFunctions.logabsbinomial(n, k)[1]

# these two compute one term of the explicit algorithm of the stirlings2. Note that they are incorrect for j = 0!
stirlings2ExplLogTerm(n::T, k::T, j::T) where T<:Integer = SpecialFunctions.logabsbinomial(k, j)[1] + n * log(j)
stirlings2ExplTerm(n::T, k::T, j::T) where T<:Integer = binomial(k, j) * j^n

function stirlings2(n::T, k::T) where T <: Integer

	res = zero(T)
	sign = -1
	for j in 1:k
		term = stirlings2ExplTerm(n, k, j)
		res += sign * term
		sign *= -1
	end
	return res ÷ factorial(k)
end

function logstirlings2(n::T, k::T) where T <: Integer

	logvalues = Vector{Float64}(undef, k)
	for j in 1:k
		logvalues[j] = stirlings2ExplLogTerm(n, k, j)
	end
	return alternating_logsumexp_batch(logvalues) - SpecialFunctions.logfactorial(k)

end

alternatingIterator(x) = Iterators.cycle(isodd(length(x) - 1) ? (-1.0, 1.0) : (1.0, -1.0))
function alternating_logsumexp_batch(x::AbstractVector{T}) where T <: Number
	# accounts for the flipping 1, -1
	# there is also a streaming version, see http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html
	alpha = maximum(x) # TODO: not sure if that's the best approach (it might introduce underflow without avoiding overflow)
	it = alternatingIterator(x)
	return log(sum((sign, value) -> sign * exp(value - alpha), zip(x, it))) + alpha
end
function logsumexp_batch(x::AbstractVector{T}) where T <: Number
	alpha = maximum(x)
	return log(sum(value -> exp(value - alpha), x)) + alpha
end

stirlings2rExplLogTerm(n, k, r, j) = logbinomial(n - r, j) + logstirlings2(j, k - r) + (n - r - j) * log(r)
stirlings2rExplTerm(n, k, r, j) = binomial(n - r, j) * stirlings2(j, k - r) * r^(n - r - j)

stirlings2r()

logstirlings2(30, 17)


exp(logstirlings2(80, 17))
stirlingsExpl(BigInt(80), BigInt(17))

logstirlings2(5, 2)
log(stirlings2(5, 2))

vv = randn(1000)
f1(x) = sign(x) * exp(log(abs(x)))
all(f1.(vv) .≈ vv)

stirlingsExpl(BigInt(30), BigInt(17))

stirlingsExpl(3, 2)
stirlings2(3, 2)

n = 5; k = 4
[stirlings2ExplLogTerm(n, k, j) for j in 1:k] ≈ [log(stirlings2ExplTerm(n, k, j)) for j in 1:k]
[log(binomial(k, j) * j^n) for j in 1:k] ≈ [SpecialFunctions.logabsbinomial(k, j)[1] + n * log(j) for j in 1:k]

stirlings2(n, k)
terms =    [stirlings2ExplTerm(n, k, j) for j in 1:k]
sum(x->x[1]*x[2], zip(terms, makeIterator(terms))) / factorial(k)

logTerms = Float64[stirlings2ExplLogTerm(n, k, j) for j in 1:k]
sum(x->exp(x[1])*x[2], zip(logTerms, makeIterator(logTerms))) / factorial(k)

alpha = 3
(exp(alpha) * sum(x->exp(x[1] - alpha) * x[2], zip(logTerms, makeIterator(logTerms))))  / factorial(k)

using LinearAlgebra

n, m = 10, 20
A = randn(n, m)
B = randn(n, n)
A' * B * A

F = LinearAlgebra.ldlt(B)


logstirlings2r

n, k, r = 4, 2, 1
sum(j->stirlings2rExplLogTerm(n, k, r, j), 0:k)
stirlings2r(n, k, r)


n, k, r = 4, 2, 1
EqualitySampler.stirlings2rExplTerm.(n, k, r, 0:k+2)
binomial(n-r, 0)
stirlings2(0, k-r)
Combinatorics.stirlings2(0, k-r)
r^(n-r-0)

stirlings2rExplTerm(n::T, k::T, r::T, j::T) where T <: Integer = binomial(n - r, j) * stirlings2(j, k - r) * r^(n - r - j)

[stirlings2r(n, k, 1) for n in 1:6, k in 1:6]

n-r
sum(j->EqualitySampler.stirlings2rExplTerm(n, k, r, j), 1:k+1)