# TODO: the calls to make_nk in a loop could be a bit more efficient

struct RandomProcessDistribution{RPM<:Turing.RandomMeasures.AbstractRandomProbabilityMeasure, T<:Integer} <: Distributions.DiscreteMultivariateDistribution
	rpm::RPM
	k::T
end

Base.eltype(::RandomProcessDistribution{RPM, T}) where {RPM<:Turing.RandomMeasures.AbstractRandomProbabilityMeasure, T<:Integer} = T

length(D::RandomProcessDistribution) = D.k

expected_model_probabilities(D::RandomProcessDistribution) = expected_model_probabilities(D, generate_distinct_models(length(D)))
function expected_model_probabilities(D::RandomProcessDistribution, models::AbstractMatrix{T}) where T <: Integer
	no_models = size(models)[2]
	modelProbs = ones(Float64, no_models)
	for i in 1:no_models, j in 2:length(D)
		modelProbs[i] *= Distributions.pdf(Turing.RandomMeasures.ChineseRestaurantProcess(D.rpm, make_nk(view(models, 1:j-1, i))), models[j, i])
	end
	modelProbs
end

expected_inclusion_probabilities(D::RandomProcessDistribution) = expected_inclusion_probabilities(D, generate_distinct_models(length(D)))
function expected_inclusion_probabilities(D::RandomProcessDistribution, models::AbstractMatrix{T}) where T <: Integer
	probs = expected_model_probabilities(D, models)
	result = OrderedDict{Int, Float64}()
	sizehint!(result, length(D))
	for i in eachindex(probs)
		key = count_equalities(view(models, :, i))
		if haskey(result, key)
			result[key] += probs[i]
		else
			result[key] = probs[i]
		end
	end
	reverse(collect(values(result)))
end

function make_nk(z)
	K = maximum(z)
	nk = Vector{Int}(map(k -> sum(z .== k), 1:K))
	nk
end

function Distributions._logpdf(d::RandomProcessDistribution, x::AbstractArray{T}) where T<:Integer
	logprob = zeros(Float64, size(x)[2])
	for i in axes(x, 2)
		logprob[i] = Distributions._logpdf(d, view(x, :, i))
	end
	logprob
end

function Distributions._logpdf(d::RandomProcessDistribution, x::AbstractVector{T}) where T<:Integer

	logprob = 0.0
	for j in 2:length(d)
		logprob += Distributions.pdf(Turing.RandomMeasures.ChineseRestaurantProcess(D.rpm, make_nk(view(x, 1:j-1))), models[j, i])
	end
	logprob
end

# should also implement Distributions._rand!(::AbstractRNG, d::MultivariateDistribution, x::AbstractArray)