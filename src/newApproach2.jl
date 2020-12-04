import Distributions, Random, Combinatorics
# import Turing: TArray # not sure if we need this one
import StatsBase: countmap
import Base: length

#=

	TODO: 
		consider caching the pdf values for BetaBinomial
		transform to parametrization by other people

=#

# Helpers / other
function generate_distinct_models(k::Int)
	# based on https://stackoverflow.com/a/30898130/4917834
	# TODO: return a generator rather than directly all results
	current = ones(Int, k)
	no_models = no_distinct_models(k)
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

function reduce_model(x::Vector{T}) where T <: Integer

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

count_equalities(urns::Vector{T}) where T <: Integer = length(urns) - length(Set(urns))
count_equalities(urns::AbstractString) = length(urns) - length(Set(urns))

function parametrize_Gopalan_Berry(x::Vector{<:Integer})
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

# Abstract
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

Distributions.rand(::Random.AbstractRNG, d::ConditionalUrnDistribution) = rand(Distributions.Categorical(_pdf(d)), 1)[1]

no_distinct_models(d::ConditionalUrnDistribution) = no_distinct_models(length(d))
no_distinct_models(k::Int) = Combinatorics.bellnum(k)

Distributions.mean(d::ConditionalUrnDistribution) = sum(_pdf(d) .* d.urns)
Distributions.var(d::ConditionalUrnDistribution) = sum(_pdf(d) .* d.urns .^2) - Distributions.mean(d)^2

# Uniform
struct UniformConditionalUrnDistribution{T<:Integer} <: ConditionalUrnDistribution{T}
	# urns::Union{Vector{T}, TArray{T,1}}
	urns::Vector{T}
	index::T
end

function UniformConditionalUrnDistribution(urns::Vector{T}, index::T) where T <: Integer
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
	x = no_distinct_models(k)
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
	# urns::Union{Vector{T}, TArray{T,1}}
	urns::Vector{T}
	index::T
	α::Float64
	β::Float64
end

function BetaBinomialConditionalUrnDistribution(urns::Vector{T}, index::T, α::Float64 = 1.0, β::Float64 = 1.0) where T <: Integer
	n = length(urns)
	all(x-> one(T) <= x <= n, urns) || throw(DomainError(urns, "condition: 0 < urns[i] < length(urns) ∀i is violated"))
	one(T) <= index <= n || throw(DomainError(urns, "condition: 0 < index < length(urns) is violated"))
	0.0 <= α || throw(DomainError(α, "condition: 0 <= α is violated"))
	0.0 <= β || throw(DomainError(β, "condition: 0 <= β is violated"))
	BetaBinomialConditionalUrnDistribution{T}(urns, index, α, β)
end

function BetaBinomialConditionalUrnDistribution(urns::Vector{T}, index::T, α::Real = 1.0, β::Real = 1.0) where T <: Integer
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

