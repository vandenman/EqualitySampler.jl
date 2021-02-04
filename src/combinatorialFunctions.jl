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
	alpha = maximum(x)
	for i in eachindex(x)
		result += sign * exp(x[i] - alpha)
		sign *= -1
	end
	# TODO: come up with something better than this fallback!
	if result < 0.0
		return T(log(sum(value->exp(BigFloat(value)), x)))
	else
		return log(result) + alpha
	end
	# return log(sum((sign, value) -> sign * exp(value - alpha), zip(x, it))) + alpha
end

function logsumexp_batch(x::AbstractVector{T}) where T <: Number
	isone(length(x)) && return x[1]
	alpha = maximum(x)
	return log(sum(value -> exp(value - alpha), x)) + alpha
end

#endregion

#region Stirling numbers of the second kind

# these two compute one term of the explicit algorithm of the stirlings2. Note that they are incorrect for j = 0!
stirlings2ExplLogTerm(n::T, k::T, j::T) where T<:Integer = SpecialFunctions.logabsbinomial(k, j)[1] + n * log(j)
stirlings2ExplTerm(n::T, k::T, j::T) where T<:Integer = binomial(k, j) * j^n

"""
	stirlings2(n::T, k::T) where T <: Integer
	stirlings2(n::T, k::T, ::Type{ExplicitStrategy})  where T <: Integer
	stirlings2(n::T, k::T, ::Type{RecursiveStrategy}) where T <: Integer

Compute the Stirlings numbers of the second kind.
The `ExplicitStrategy` (default) uses an explicit loop and is computationally more efficient but subject to overflow, so using BigInt is advised.
The `RecursiveStrategy` uses recursion and is mathematically elegant yet inefficient for large values.
"""
stirlings2(n::T, k::T) where T <: Integer = stirlings2(n, k, ExplicitStrategy)
function stirlings2(n::T, k::T, ::Type{ExplicitStrategy}) where T <: Integer

	# TODO: do something nicer with these common cases
	succes, value = stirlings2_base_cases(n, k)
	succes && return value

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

# required for memoize
Memoize.@memoize function stirlings2_recursive(n::T, k::T) where T <: Integer

	succes, value = stirlings2_base_cases(n, k)
	succes && return value

	return k * stirlings2_recursive(n - 1, k) + stirlings2_recursive(n - 1, k - 1)
end

"""
Base cases for stirlings2. Should return a tuple of (base_case_found::Bool, value::T).
"""
function stirlings2_base_cases(n::T, k::T) where T <: Integer

	if n < zero(T)
		throw(DomainError(n, "n must be nonnegative"))
	elseif k > n
		return (true, zero(T))
	elseif n == k
		return (true, one(T))
	elseif n == zero(T) || k == zero(T)
		return (true, zero(T))
	elseif k == n - 1
		return (true, binomial(n, T(2)))
	elseif k == T(2)
		return (true, 2^(n-1) - 1)
	end
	return (false, zero(T))

end

"""
	logstirlings2(n::T, k::T) where T <: Integer

Compute the logarithm of the Stirlings numbers of the second kind with an explicit formula.
"""
function logstirlings2(n::T, k::T) where T <: Integer

	succes, value = stirlings2_base_cases(n, k)
	succes && value >= zero(T) && return log(Float64(value))

	logvalues = Vector{Float64}(undef, k)
	for j in 1:k
		logvalues[j] = stirlings2ExplLogTerm(n, k, j)
	end
	return alternating_logsumexp_batch(logvalues) - SpecialFunctions.logfactorial(k)

end
#endregion

#region r-Stirling numbers of the second kind

stirlings2rExplLogTerm(n::T, k::T, r::T, j::T) where T <: Integer = logbinomial(n - r, j) + logstirlings2(j, k - r) + (n - r - j) * log(r)
stirlings2rExplTerm(n::T, k::T, r::T, j::T) where T <: Integer = binomial(n - r, j) * stirlings2(j, k - r) * r^(n - r - j)
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

	result = 0.0
	for j in 1:n-r
		result += stirlings2rExplTerm(n, k, r, j)
	end
	return result
end

stirlings2r(n::T, k::T, r::T, ::Type{RecursiveStrategy}) where T <: Integer = stirlings2r_recursive(n, k, r)

# required for memoization
Memoize.@memoize function stirlings2r_recursive(n::T, k::T, r::T) where T <: Integer

	succes, value = stirlings2r_base_cases(n, k, r)
	succes && return value
	return k * stirlings2r_recursive(n - 1, k, r) + stirlings2r_recursive(n - 1, k - 1, r)

end

"""
Base cases for stirlings2r. Should return a tuple of (base_case_found::Bool, value::T).
"""
function stirlings2r_base_cases(n::T, k::T, r::T) where T <: Integer

	n < r && return (true, zero(T))
	(k > n || k < r) && return (true, zero(T))
	n == k && return (true, one(T))
	iszero(n) || iszero(k) && return (true, zero(T))
	n == r && return (true, one(T))
	k == r && return (true, r^(n - r))

	return (false, zero(T))
end

"""
	logstirlings2r(n::T, k::T, r::T) where T <: Integer

Computes the logarithm of the r-stirling numbers with an explicit formula.
"""
function logstirlings2r(n::T, k::T, r::T) where T <: Integer

	succes, value = stirlings2r_base_cases(n, k, r)
	succes && value >= zero(T) && return log(Float64(value))

	values = Vector{Float64}(undef, n-r)
	for j in eachindex(values)
		values[j] = stirlings2rExplLogTerm(n, k, r, j)
	end
	return logsumexp_batch(values)
end

"""
	stirlings2r_recursive(n::T, k::T, r::T) where T <: Integer

Computes the r-stirling numbers with a recursive method. Uses memoization for efficiency.
"""
#endregion


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

"""
	generate_distinct_models(k::Int)

Generates all distinct models that represent equalities.
"""
function generate_distinct_models(k::Int)
	# based on https://stackoverflow.com/a/30898130/4917834
	# TODO: return a generator rather than directly all results
	current = ones(Int, k)
	no_models = count_distinct_models(k)
	result = Matrix{Int}(undef, k, no_models)
	result[:, 1] .= current
	isone(k) && return result
	range = k:-1:2
	for i in 2:no_models

		idx = findfirst(i->current[i] < k && any(==(current[i]), current[1:i-1]), range)
		rightmost_incrementable = range[idx]
		current[rightmost_incrementable] += 1
		current[rightmost_incrementable + 1 : end] .= 1
		result[:, i] .= current

	end
	return result
end


