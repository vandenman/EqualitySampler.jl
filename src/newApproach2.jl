import Distributions, Random, Combinatorics
# import Turing: TArray # not sure if we need this one
import StatsBase: countmap
import Base: length

#=

	TODO: 
		consider caching the pdf values for BetaBinomial
		transform to parametrization by other people

		Consider this:
			p(x[1], x[2], x[3]) = p(x[1]) * p(x[2] | x[1]) * p(x[3] | x[2], x[1])
			that's how the uniform multivariate sampler works, and that works fine.
			relabeling can also be done during that algorithm already.
			If we can figure out how to do the same with the BetaBinomial we should be golden.

			Also study the DirichletMultinomial distribution a little bit more.


=#

# Helpers / other
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

"""
	reduce a model to a unique representation. For example, [2, 2, 2] -> [1, 1, 1]
"""
function reduce_model(x::AbstractVector{T}) where T <: Integer

	y = copy(x)
	for i in eachindex(x)
		if !any(==(x[i]), x[1:i - 1])
			if x[i] != i
				idx = findall(==(x[i]), x[i:end]) .+ (i - 1)
				y[idx] .= i
			end
		end
	end
	return y
end

count_combinations(k, islands) = factorial(islands) * binomial(k, islands)
function count_combinations(s::AbstractString)
	s = filter(!isspace, s)
	k = length(s)
	islands = length(unique(s))
	return count_combinations(k, islands)
end

count_combinations(x::AbstractVector) = count_combinations(length(x), length(unique(x)))

count_equalities(urns::AbstractVector{T}) where T <: Integer = length(urns) - length(Set(urns))
count_equalities(urns::AbstractString) = length(urns) - length(Set(urns))

function parametrize_Gopalan_Berry(x::AbstractVector{T}) where T <: Integer
	C = zero(x)
	table = countmap(values(countmap(x)))
	for (k, v) in table
		C[k] = v
	end
	return C
end

# are the two below even used?
function counts2probs(counts::Dict{T, Int}) where T
	total_visited = sum(values(counts))
	probs = Dict{T, Float64}()
	for (model, count) in counts
		probs[model] = count / total_visited
	end
	return probs
end

function count_equalities(sampled_models)

	k, n = size(sampled_models)
	result = Vector{Int}(undef, n)
	for i in axes(sampled_models, 2)
		result[i] = k - length(unique(view(sampled_models, :, i)))
	end
	return result
end

# Abstract ----
abstract type ConditionalUrnDistribution{T} <: Distributions.DiscreteUnivariateDistribution where T <: Integer end

Distributions.minimum(d::ConditionalUrnDistribution{T}) where T = one(T)
Distributions.maximum(d::ConditionalUrnDistribution{T}) where T = T(length(d))
Base.length(d::ConditionalUrnDistribution) = length(d.urns)

Distributions.logpdf(d::ConditionalUrnDistribution, x::Real) = -Inf
Distributions.pdf(d::ConditionalUrnDistribution{T}, x::Real) where T = exp(logpdf(d, x))
function Distributions.logpdf(d::ConditionalUrnDistribution{T}, x::T) where T <: Integer
	Distributions.insupport(d, x) || return -Inf
	return log(_pdf(d)[x])
end

Distributions.cdf(d::ConditionalUrnDistribution, x::Real) = 0.0
function Distributions.cdf(d::ConditionalUrnDistribution, x::T) where T <: Integer
	cdf = cumsum(_pdf(d))
	return cumsum[x]
end

Distributions.rand(::Random.AbstractRNG, d::ConditionalUrnDistribution) = rand(Distributions.Categorical(_pdf(d)), 1)[1]

count_distinct_models(d::ConditionalUrnDistribution) = count_distinct_models(length(d))
count_distinct_models(k::Int) = Combinatorics.bellnum(k)
count_models_with_incl(k, no_equalities) = Combinatorics.stirlings2(k, k - no_equalities) .* count_combinations.(k, k - no_equalities)


Distributions.mean(d::ConditionalUrnDistribution) = sum(_pdf(d) .* d.urns)
Distributions.var(d::ConditionalUrnDistribution) = sum(_pdf(d) .* d.urns .^2) - Distributions.mean(d)^2

# Uniform
struct UniformConditionalUrnDistribution{T<:Integer} <: ConditionalUrnDistribution{T}
	# urns::Union{Vector{T}, TArray{T,1}}
	urns::AbstractVector{T}
	index::T
end

function UniformConditionalUrnDistribution(urns::AbstractVector{T}, index::T) where T <: Integer
	n = length(urns)
	all(x-> one(T) <= x <= n, urns) || throw(DomainError(urns, "condition: 0 < urns[i] < length(urns) ∀i is violated"))
	one(T) <= index <= n || throw(DomainError(urns, "condition: 0 < index < length(urns) is violated"))
	UniformConditionalUrnDistribution{T}(urns, index)
end

function _pdf(d::UniformConditionalUrnDistribution)

	urns, index = d.urns, d.index
	
	oldvalue = urns[index]
	counts = Vector{Int}(undef, length(urns))
	for i in eachindex(counts)
		urns[index] = i
		counts[i] = count_combinations(urns)
	end
	urns[index] = oldvalue

	counts = 1 ./ counts
	counts = counts ./ sum(counts)
	return counts
end

function expected_model_probabilities(k::Int)
	x = count_distinct_models(k)
	return fill(1 / x, x)
end
expected_model_probabilities(d::UniformConditionalUrnDistribution) = expected_model_probabilities(length(d))


expected_inclusion_counts(k::Int) = Combinatorics.stirlings2.(k, k:-1:1)
function expected_inclusion_probabilities(k::Int)
	counts = expected_inclusion_counts(k)
	return counts ./ sum(counts)
end
expected_inclusion_counts(d::UniformConditionalUrnDistribution) = expected_inclusion_counts(length(d))
expected_inclusion_probabilities(d::UniformConditionalUrnDistribution) = expected_inclusion_probabilities(length(d))

# BetaBinomial
struct BetaBinomialConditionalUrnDistribution{T<:Integer} <: ConditionalUrnDistribution{T}
	# urns::Union{AbstractVector{T}, TArray{T,1}}
	urns::AbstractVector{T}
	index::T
	α::Float64
	β::Float64
end

function BetaBinomialConditionalUrnDistribution(urns::AbstractVector{T}, index::T, α::Float64 = 1.0, β::Float64 = 1.0) where T <: Integer
	n = length(urns)
	all(x-> one(T) <= x <= n, urns) || throw(DomainError(urns, "condition: 0 < urns[i] < length(urns) ∀i is violated"))
	one(T) <= index <= n || throw(DomainError(urns, "condition: 0 < index < length(urns) is violated"))
	0.0 <= α || throw(DomainError(α, "condition: 0 <= α is violated"))
	0.0 <= β || throw(DomainError(β, "condition: 0 <= β is violated"))
	BetaBinomialConditionalUrnDistribution{T}(urns, index, α, β)
end

function BetaBinomialConditionalUrnDistribution(urns::AbstractVector{T}, index::T, α::Real = 1.0, β::Real = 1.0) where T <: Integer
	BetaBinomialConditionalUrnDistribution{T}(urns, index, convert(Float64, α), convert(Float64, β))
end


function _pdf(d::BetaBinomialConditionalUrnDistribution)

	k = length(d)
	expected_counts = expected_inclusion_counts(k)
	newinclusions = Vector{Int}(undef, k)
	urns = d.urns
	index = d.index
	oldvalue = urns[index]
	# count how many equalities there would be for each possible k
	for i in 1:k
		urns[index] = i
		newinclusions[i] = count_equalities(urns)
	end
	urns[index] = oldvalue

	# count how many models there are with each number of inclusions
	newcounts = expected_counts[newinclusions .+ 1]

	# account for duplicate models
	newcounts = newcounts .* count_combinations.(k, length(urns) .- newinclusions)

	# normalize to probabilities
	newcounts = 1 ./ newcounts
	newcounts = newcounts ./ sum(newcounts)

	# reweight by BetaBinomial weights
	newcounts = newcounts .* Distributions.pdf.(Distributions.BetaBinomial(k - 1, d.α, d.β), newinclusions)
	newcounts = newcounts ./ sum(newcounts)

	return newcounts

end

function expected_inclusion_probabilities(d::BetaBinomialConditionalUrnDistribution)
	k = length(d) - 1
	return Distributions.pdf(Distributions.BetaBinomial(k, d.α, d.β), 0:k)
end

function expected_model_probabilities(d::BetaBinomialConditionalUrnDistribution)
	incl_probs  = expected_inclusion_probabilities(d)
	no_models_with_incl = expected_inclusion_counts(length(d))
	
	# probability of j equalities for j in 1...k
	probs = incl_probs ./ no_models_with_incl
	result = Vector{Float64}(undef, sum(no_models_with_incl))
	index = 1
	for i in eachindex(probs)
		result[index:index + no_models_with_incl[i] - 1] .= probs[i]
		index += no_models_with_incl[i]
	end
	return result
end

# Multivariate versions

abstract type AbstractMvUrnDistribution{T <: Integer} <: Distributions.DiscreteMultivariateDistribution end

Base.length(d::AbstractMvUrnDistribution) = d.k
Base.eltype(::AbstractMvUrnDistribution{T}) where T = T

# TODO: implement this!
# sampler(d::Distribution)


struct UniformMvUrnDistribution{T} <: AbstractMvUrnDistribution{T}
	k::Int
end
function UniformMvUrnDistribution(k::T) where T <: Integer
	k > zero(T) || throw(DomainError(k, "k must be larger than zero"))
	return UniformMvUrnDistribution{Int}(k)
end

function Distributions._logpdf(d::UniformMvUrnDistribution{T}, x::AbstractVector{T}) where T

	# probability of the unique version of that model
	log_prob_of_model = -log(count_distinct_models(length(d)))
	# divide by the number of duplicate models
	log_prob_of_model -= log(count_combinations(x))
	return log_prob_of_model

end

function Distributions.logpdf(d::AbstractMvUrnDistribution{T}, x::T) where T <: Integer

	!Distributions.insupport(d, x) || return -Inf
	return Distributions._logpdf(d, x)

end

function Distributions._rand!(rng::Random.AbstractRNG, d::AbstractMvUrnDistribution, x::AbstractMatrix)
	for i in axes(x, 2)
		Distributions._rand!(rng, d, view(x, :, i))
	end
	x
end

function BN(n::T, r::T) where T <: Integer

	res = zero(T)
	for k in 0:n, i in 0:n
		res +=
			binomial(n, i) *
			Combinatorics.stirlings2(i, k) *
			r^(n - i)
	end
	return res
end

function get_conditional_counts(n::Int, known::AbstractVector{T} = [1]) where T <: Integer

	# TODO: just pass d.partitions and d.index to this function! (or just d itself)
	# or maybe not?
	refinement = length(unique(known))
	n_known = length(known)

	res = zeros(Int, n_known + 1)
	idx = findall(i->known[i] == i, 1:n_known)
	n_idx = length(idx)
	res[idx] .= BN(n - n_known - 1, n_idx)
	res[n_known + 1] = BN(n - n_known - 1, n_idx + 1)
	res
end

function Distributions._rand!(::Random.AbstractRNG, d::UniformMvUrnDistribution, x::AbstractVector)

	k = d.k
	x[1] = 1
	for j in 2:k
		count = get_conditional_counts(k, x[1:j-1])
		x[j] = rand(Distributions.Categorical(count ./ sum(count)), 1)[1]
	end
	_relabel!(x)
	x
end

function _relabel!(x)
	k = length(x)
	new_x = Random.randperm(k)[x]
	x .= new_x
end

expected_model_probabilities(d::UniformMvUrnDistribution) = expected_model_probabilities(length(d))
expected_inclusion_counts(d::UniformMvUrnDistribution) = expected_inclusion_counts(length(d))
expected_inclusion_probabilities(d::UniformMvUrnDistribution) = expected_inclusion_probabilities(length(d))

struct BetaBinomialMvUrnDistribution{T} <: AbstractMvUrnDistribution{T}
	k::Int
	α::Float64
	β::Float64
end

function BetaBinomialMvUrnDistribution(k::T, α::Real = 1.0, β::Real = 1.0) where T <: Integer
	BetaBinomialMvUrnDistribution{T}(k, convert(Float64, α), convert(Float64, β))
end


function Distributions._logpdf(d::BetaBinomialMvUrnDistribution, urns::AbstractArray)
	
	k = length(d)
	no_incl = count_equalities(urns)
	prob_of_no_incl = Distributions.logpdf(Distributions.BetaBinomial(k - 1, d.α, d.β), no_incl)
	no_models_with_incl = count_combinations(k, k - no_incl)
	return prob_of_no_incl / no_models_with_incl
end

function expected_model_probabilities(d::BetaBinomialMvUrnDistribution)
	incl_probs  = expected_inclusion_probabilities(d)
	no_models_with_incl = expected_inclusion_counts(length(d))
	
	# probability of j equalities for j in 1...k
	probs = incl_probs ./ no_models_with_incl
	result = Vector{Float64}(undef, sum(no_models_with_incl))
	index = 1
	for i in eachindex(probs)
		result[index:index + no_models_with_incl[i] - 1] .= probs[i]
		index += no_models_with_incl[i]
	end
	return result
end

function expected_inclusion_probabilities(d::BetaBinomialMvUrnDistribution) 
	k = length(d) - 1
	return Distributions.pdf(Distributions.BetaBinomial(k, d.α, d.β), 0:k)
end

function Distributions._rand!(::Random.AbstractRNG, d::BetaBinomialMvUrnDistribution, x::AbstractVector)

	# TODO: this does not work yet!
	k = length(d)
	x[1] = rand(1:4, 1)[1]
	inclusion_probs = expected_inclusion_probabilities(d)
	for j in 2:k
		count = get_conditional_counts(k, x[1:j-1])

		probs = count ./ sum(count)

		# count the number of inclusions for each result
		no_incl = zeros(Int, j)
		v = view(x, 1:j)
		for i in 1:j
			x[j] = i
			no_incl[i] = count_equalities(v)
		end

		probs .*= inclusion_probs[no_incl .+ 1]
		probs ./= sum(probs)

		x[j] = rand(Distributions.Categorical(probs), 1)[1]
	end
	_relabel!(x)
	x
end


function draw_test_bernoulli(s, k)

	iszero(s) && return zeros(Int, k)
	s == k && return ones(Int, k)
	result = zeros(Int, k)
	result[1] = rand(0:1)
	for i in 2:k
		prob = (s - sum(result[1:i-1])) / (k - i + 1)
		result[i] = rand() <= prob ? 1 : 0
	end
	result
end

rand(UniformMvUrnDistribution(3), 1)

draw_test_bernoulli(2, 3)

function draw_test_categorical(target_incl, k)

	iszero(target_incl) && return collect(1:k)
	target_incl == k && return ones(Int, k)

	result = zeros(Int, k)
	result[1] = rand(1:k)

	prob = Vector{Float64}(undef, k)
	current_incl = 0
	for i in 2:k

		remaining_spots = k - i + 1
		probnew = (target_incl - current_incl) / remaining_spots
		if rand() <= probnew
			current_incl += 1
			result[i] = rand(result[1:i-1], 1)[1]
		else
			candidates = setdiff(1:k, result)
			result[i] = rand(candidates, 1)[1]
		end
	end
	return result
end

function count_islands(x::AbstractVector{T}) where T<: Integer
	sort(countmap(x))
end

nsim = 1_000_000
target = 2
k = 5
mat = Matrix{Int}(undef, k, nsim);
for i in 1:nsim
	mat[:, i] .= reduce_model(draw_test_categorical(2, k))
end

islands = vec(mapslices(x->maximum(values(count_islands(x))), mat, dims = 1))
countmap(islands)

# # rand(UniformMvUrnDistribution(3), 1)

# function stirling2_parts(n, k, i)
# 	(-1)^i * binomial(k, i) * (k - i)^n
# end
# stirling2_parts.(2, 1, 0:2)
# Combinatorics.stirlings2(5, 3)
# sum(stirling2_parts.(5, 3, 0:2)) ÷ factorial(3)

# function stirling2_man(n, k)
# 	ans = 0
# 	for i in 0:k-1
# 		ans += (-1)^i * binomial(k, i) * (k - i)^n
# 	end
# 	 ans ÷ factorial(k > 10 ? big(k) : k)
# end

k = 3
urns = ones(Int, k)
urns = [1, 2, 3]
Dall  = UniformMvUrnDistribution(k)
Dcond = UniformConditionalUrnDistribution(ones(Int, k), 1)
tmp = copy(urns)
probAll  = Vector{Float64}(undef, k)
probCond = zeros(Float64, k)
for i in eachindex(tmp)
	tmp[1] = i
	probAll[i] = Distributions.pdf(Dall, tmp)
end
probCond = probAll / sum(probAll)
marginalProbs = probAll ./ probCond

urns = ones(Int, k)
urns = [1, 2, 3]
Dall  = UniformMvUrnDistribution(k)
tmp = copy(urns)
probAll  = Vector{Float64}(undef, k)
probCond = zeros(Float64, k)
for i in eachindex(tmp)
	tmp[2] = i
	probAll[i] = Distributions.pdf(Dall, tmp)
end
probAll ./ marginalProbs


Distributions.pdf(Dall, tmp)


# TODO: Distributions.pdf(UniformMvUrnDistribution) does not sum to one! Fix this!
# actually.. it does?
#=
k = 4
Dall = UniformMvUrnDistribution(k)
it = Iterators.product(fill(1:k, k)...)
totprob = 0.0
for i in it
	totprob += Distributions.pdf(Dall, collect(i))
end
totprob

mods = generate_distinct_models(k)
totprob2 = 0.0
for i in axes(mods, 2)
	totprob2 += Distributions.pdf(Dall, mods[:, i])
end

function brute_force_conditional_probs(target::Int, k::Int, known_indices::AbstractVector{T}, known_values::AbstractVector{T}, D::AbstractMvUrnDistribution) where T <: Integer

	ranges = fill(one(T):k, k)
	ranges[known_indices] .= UnitRange.(known_values, known_values)
	It = Iterators.product(ranges...)
	probs = zeros(Float64, k)
	for it in It
		arr = collect(it)
		probs[arr[target]] += Distributions.pdf(D, arr)
	end
	return probs ./ sum(probs)
end

D = UniformMvUrnDistribution(3)
brute_force_conditional_probs(3, 3, [1, 2], [1, 1], D)
get_conditional_counts(3, [1])

3*Distributions.pdf(D, [1, 1, 1])

D = BetaBinomialMvUrnDistribution(3)
brute_force_conditional_probs(2, 3, [1], [1], D)

pdf(BetaBinomialMvUrnDistribution, )



#=
	TODO II: see if we can get the conditional probabilities
	by looking at the inclusion probabilities
	so given x[1] = 1 we compute 
	p(x[2] | x[1]) by looking at p(x[2] == x[1]) vs p(x[2] != x[1])

=# 

k = 4
It = Iterators.product(fill(1:k, k)...)
dd = sort(Dict{Int, Int}(0:k-1 .=> 0))
for it in It
	no_incl = count_equalities(collect(it))
	dd[no_incl] += 1
end

mods = generate_distinct_models(k)
dd0 = sort(Dict{Int, Int}(0:k-1 .=> 0))
for col in eachcol(mods)
	no_incl = count_equalities(col)
	dd0[no_incl] += 1
end


collect(values(dd0)) .* count_combinations.(k, k:-1:1)
dd

count_models_with_incl.(k, 0:k-1)

=#