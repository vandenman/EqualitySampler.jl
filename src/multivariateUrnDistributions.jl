

#region AbstractMvUrnDistribution
abstract type AbstractMvUrnDistribution{T} <: Distributions.DiscreteMultivariateDistribution where T <: Integer end

Base.length(d::AbstractMvUrnDistribution) = d.k
Distributions.minimum(::AbstractMvUrnDistribution{T}) where T = one(T)
Distributions.maximum(d::AbstractMvUrnDistribution{T}) where T = T(length(d))

Distributions.logpdf(::AbstractMvUrnDistribution, ::AbstractVector{T}) where T <: Real = -Inf
Distributions.pdf(::AbstractMvUrnDistribution, ::AbstractVector{T}) where T <: Real = zero(T)
Distributions.pdf(d::AbstractMvUrnDistribution, x::AbstractVector{<:Integer}) = exp(Distributions.logpdf(d, x))

Distributions.eltype(::AbstractMvUrnDistribution{T}) where T = T

in_eqsupport(d::AbstractMvUrnDistribution, no_equalities::Integer) = 0 <= no_equalities < length(d)

function Distributions._rand!(rng::Random.AbstractRNG, d::AbstractMvUrnDistribution, x::AbstractArray{T,2} where T)
	for i in axes(x, 2)
		Distributions._rand!(rng, d, view(x, :, i))
	end
	x
end

function Distributions._rand!(rng::Random.AbstractRNG, d::AbstractMvUrnDistribution, x::AbstractVector)

	probvec = zeros(Float64, length(x))

	for i in eachindex(x)

		_pdf_helper!(probvec, d, i, x)
		x[i] = rand(rng, Distributions.Categorical(probvec))

	end
	x
end

function logpdf_incl(d::AbstractMvUrnDistribution, no_equalities::T) where T<:Integer
	k = length(d)
	if !(0 <= no_equalities < k)
		T <: BigInt && return BigFloat(-Inf)
		return -Inf
	end
	logpdf_model_distinct(d, no_equalities) + log_count_distinct_models_with_incl(k, no_equalities)# + log_count_combinations(k, k - no_equalities)
end

pdf_incl(d::AbstractMvUrnDistribution,  no_equalities) = exp(logpdf_incl(d,  no_equalities))
# pdf_model(d::AbstractMvUrnDistribution, no_equalities) = exp(logpdf_model(d, no_equalities))

function logpdf_model(d::AbstractMvUrnDistribution, x::T) where T <: Integer
	if x > length(d)
		T <: BigInt && return BigFloat(-Inf)
		return -Inf
	end
	logpdf_model_distinct(d, x) - log_count_combinations(length(d), length(d) - x)
end
logpdf_model(d::AbstractMvUrnDistribution, x::AbstractVector{T}) where T <: Integer = logpdf_model_distinct(d, x) - log_count_combinations(x)
pdf_model(d::AbstractMvUrnDistribution, x) = exp(logpdf_model(d, x))
pdf_model_distinct(d::AbstractMvUrnDistribution, x) = exp(logpdf_model_distinct(d, x))

Distributions.logpdf(d::AbstractMvUrnDistribution, x::AbstractVector{T}) where T<:Integer = logpdf_model(d, x)

#endregion

#region UniformMvUrnDistribution
struct UniformMvUrnDistribution{T <: Integer} <: AbstractMvUrnDistribution{T}
	k::T
end

Distributions.logpdf(d::UniformMvUrnDistribution, x::AbstractVector{T}) where T<:Integer = logpdf_model(d, x)

logpdf_model_distinct(d::UniformMvUrnDistribution, ::AbstractVector{T}) where T <: Integer = -logbellnumr(convert(T, length(d)), 0)
logpdf_model_distinct(d::UniformMvUrnDistribution, ::T) where T <: Integer = -logbellnumr(convert(T, length(d)), 0)



#endregion

#region BetaBinomialMvUrnDistribution
struct BetaBinomialMvUrnDistribution{T <: Integer} <: AbstractMvUrnDistribution{T}
	k::T
	α::Float64
	β::Float64
	_log_model_probs_by_incl::Vector{Float64}
	function BetaBinomialMvUrnDistribution(k::T, α::Float64 = 1.0, β::Float64 = 1.0) where T<:Integer
		log_model_probs_by_incl = Distributions.logpdf.(Distributions.BetaBinomial(k - 1, α, β), 0:k - 1) .- log_expected_inclusion_counts(k)
		new{T}(k, α, β, log_model_probs_by_incl)
	end
end

function BetaBinomialMvUrnDistribution(k::Integer, α::Number, β::Number = 1.0)
	BetaBinomialMvUrnDistribution(k, convert(Float64, α), convert(Float64, β))
end


# function Distributions.logpdf(d::BetaBinomialMvUrnDistribution, x::AbstractVector{<:Integer})
# 	log_model_probs_by_incl(d)[count_equalities(x) + 1]
# 	# Distributions.logpdf(Distributions.BetaBinomial(length(d) - 1, d.α, d.β), count_equalities(x))
# end

log_model_probs_by_incl(d::BetaBinomialMvUrnDistribution) = d._log_model_probs_by_incl
logpdf_model_distinct(d::BetaBinomialMvUrnDistribution, x::AbstractVector{<:Integer}) = logpdf_model_distinct(d, count_equalities(x))
function logpdf_model_distinct(d::BetaBinomialMvUrnDistribution, no_equalities::Integer)
	in_eqsupport(d, no_equalities) || return -Inf
	log_model_probs_by_incl(d)[no_equalities + 1]
end

function logpdf_incl(d::BetaBinomialMvUrnDistribution, no_equalities::Integer)
	in_eqsupport(d, no_equalities) || return -Inf
	Distributions.logpdf.(Distributions.BetaBinomial(length(d) - 1, d.α, d.β), no_equalities)
end

#endregion

struct RandomProcessMvUrnDistribution{RPM <: Turing.RandomMeasures.AbstractRandomProbabilityMeasure, T <: Integer} <: AbstractMvUrnDistribution{T}
	k::T
	rpm::RPM
end
rpm(d::RandomProcessMvUrnDistribution) = d.rpm

"""
DirichletProcessMvUrnDistribution(k::Integer, α::Float64)
DirichletProcessMvUrnDistribution(k::Integer, ::Symbol = :Gopalan_Berry)
DirichletProcessMvUrnDistribution(k::Integer, α::Real)

Wrapper function to create an object representing a Dirichlet process prior. These call RandomProcessMvUrnDistribution but are a bit more user friendly.
Either set α directly by passing a float, or pass (any) symbol to use `dpp_find_α` to specify α, which uses the heuristic by
Gopalan & Berry (1998) so that p(everything equal) == p(everything unequal).
"""
DirichletProcessMvUrnDistribution(k::Integer, α::Float64) = RandomProcessMvUrnDistribution(k, Turing.RandomMeasures.DirichletProcess(α))
DirichletProcessMvUrnDistribution(k::Integer, ::Symbol = :Gopalan_Berry) = DirichletProcessMvUrnDistribution(k, dpp_find_α(k))
DirichletProcessMvUrnDistribution(k::Integer, α::Real) = DirichletProcessMvUrnDistribution(k, convert(Float64, α))
# whats the best way to do this ↓ ?
# const DirichletProcessMvUrnDistributionType{T} = RandomProcessMvUrnDistribution{Turing.RandomMeasures.DirichletProcess{Float64}, T}

function Distributions._rand!(rng::Random.AbstractRNG, d::RandomProcessMvUrnDistribution, x::AbstractVector)
	_sample_process!(rng, rpm(d), x)
end

get_process(rpm::RandomMeasures.AbstractRandomProbabilityMeasure, ::Vector{Int}) = rpm
get_process(rpm::RandomMeasures.PitmanYorProcess, nk::Vector{Int}) = RandomMeasures.PitmanYorProcess(rpm.d, rpm.θ, sum(!iszero, nk))

# TODO: there has to be a more efficient way to compute nk!
function _sample_process!(rng::Random.AbstractRNG, rpm::RandomMeasures.AbstractRandomProbabilityMeasure, x::AbstractVector)

	x[1] = 1
	for i in 2:length(x)
		# Number of observations per cluster.

		K = maximum(view(x, 1:i-1))
		# nk = Vector{Int}(map(k -> sum(x .== k), 1:K))
		vx = view(x, 1:i-1)
		K = maximum(vx)
		nk = Vector{Int}(map(k -> sum(==(k), vx), 1:K))

		# Draw new assignment.
		x[i] = rand(rng, RandomMeasures.ChineseRestaurantProcess(get_process(rpm, nk), nk))
	end
	x
end

# Distributions.logpdf(::RandomProcessMvUrnDistribution, ::AbstractVector{T}) where T<: Real = -Inf
# function Distributions.logpdf(d::RandomProcessMvUrnDistribution, x::AbstractVector{T}) where T<: Integer

# 	lpdf = zero(Float64)
# 	K = x[1]
# 	for i in 2:length(x)

# 		vx = view(x, 1:i-1)
# 		nk = Vector{Int}(map(k -> sum(==(k), vx), 1:K))
# 		lpdf += Distributions.logpdf(Turing.RandomMeasures.ChineseRestaurantProcess(d.rpm, nk), x[i])
# 		K = max(K, x[i])

# 		# vnk = view(nk, 1:K)
# 	end
# 	lpdf
# end

function logpdf_incl(d::RandomProcessMvUrnDistribution, no_equalities::Integer)

	throw(DomainError(d, "only implemented for rpm<:Turing.RandomMeasures.DirichletProcess"))
	# generate one instance of all different models with no_equalities

	# result = zero(Float64)
	# opts = generate_distinct_models_with_sum(length(d), no_equalities)
	# for i in axes(opts, 2)
	# 	result += logpdf(d, view(opts, :, i)) + count_distinct_models_with_pattern(opts)
	# end
	# return result
end

function logpdf_model_distinct(d::RandomProcessMvUrnDistribution{W, T}, x::U) where {T<:Integer, U<:Integer, W}
	logpdf_model_distinct(d::RandomProcessMvUrnDistribution{W, T}, convert(T, x))
end

function logpdf_model_distinct(d::RandomProcessMvUrnDistribution{W, T}, x::T) where {T<:Integer, W}

	U = T <: BigInt ? BigFloat : Float64
	n = T(length(d))
	M = U(d.rpm.α)

	# idx = findall(==(n - x), length.(sizes))
	target_size = n - x
	f! = x->filter!(y->length(y) == target_size, x)
	counts, sizes = count_set_partitions_given_partition_size(f!, n)

	res = U(0.0)
	for i in eachindex(counts)#idx
		v = length(sizes[i]) * log(M) +
			SpecialFunctions.logabsgamma(M)[1] -
			SpecialFunctions.logabsgamma(M + n)[1] +
			sum(x->SpecialFunctions.logabsgamma(x)[1], sizes[i])

		res += counts[i] * v
	end
	return res / sum(counts)
end

function logpdf_model_distinct(d::RandomProcessMvUrnDistribution{RPM, T}, x::AbstractVector{<:Integer})  where {RPM<:RandomMeasures.DirichletProcess, T<:Integer}

	n = length(d)
	M = d.rpm.α
	cc = countmap(x)

	return length(cc) * log(M) +
		SpecialFunctions.logabsgamma(M)[1] -
		SpecialFunctions.logabsgamma(M + n)[1] +
		sum(x->SpecialFunctions.logabsgamma(x)[1], values(cc))

end

function logpdf_incl(d::RandomProcessMvUrnDistribution{RPM, T}, no_equalities::Integer) where {RPM<:RandomMeasures.DirichletProcess, T<:Integer}

	#=
		From chapter 3: Dirichlet Process, equation 3.6
		Peter Müller, Abel Rodriguez
		NSF-CBMS Regional Conference Series in Probability and Statistics, 2013: 23-41 (2013) https://doi.org/10.1214/cbms/1362163748

		In 3.6 the extra n! term cancels when normalizing to a probability
	=#

	n = length(d)
	M = d.rpm.α
	k = n - no_equalities # number of unique values

	logunsignedstirlings1(n, k) +
		# SpecialFunctions.logfactorial(n) +
		k * log(M) +
		SpecialFunctions.logabsgamma(M)[1] -
		SpecialFunctions.logabsgamma(M + n)[1]

end

expected_inclusion_probabilities(d::RandomProcessMvUrnDistribution) = [pdf_incl(d, i) for i in 0:length(d) - 1]

# TODO: this method doesn't work yet
# function expected_model_probabilities(d::RandomProcessMvUrnDistribution)

# 	incl_probs  = expected_inclusion_probabilities(d)
# 	no_models_with_incl = expected_inclusion_counts(length(d))
# 	probs = incl_probs ./ no_models_with_incl

# 	# TODO: this compact creates type instabilities!
# 	if compact

# 		result = hcat(0:length(d)-1, no_models_with_incl, probs)

# 	else

# 		# probability of j equalities for j in 1...k
# 		result = Vector{Float64}(undef, sum(no_models_with_incl))
# 		index = 1
# 		for i in eachindex(probs)
# 			result[index:index + no_models_with_incl[i] - 1] .= probs[i]
# 			index += no_models_with_incl[i]
# 		end
# 	end
# 	return result

# end