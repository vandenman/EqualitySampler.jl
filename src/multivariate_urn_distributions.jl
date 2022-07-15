#= NOTES

	*_distinct	does NOT account for duplicate configurations. For example, [1, 1, 1] is     the only null model and [2, 2, 2] does NOT exist
	*			does     account for duplicate configurations. For example, [1, 1, 1] is NOT the only null model and [2, 2, 2] does     exist (and so does [3, 3, 3])

=#



#region AbstractMvUrnDistribution
abstract type AbstractMvUrnDistribution{T} <: Distributions.DiscreteMultivariateDistribution where T <: Integer end

Base.length(d::AbstractMvUrnDistribution) = d.k
Distributions.minimum(::AbstractMvUrnDistribution{T}) where T = one(T)
Distributions.maximum(d::AbstractMvUrnDistribution{T}) where T = T(length(d))

Distributions.logpdf(::AbstractMvUrnDistribution, ::AbstractVector{T}) where T <: Real = -Inf
Distributions.pdf(::AbstractMvUrnDistribution, ::AbstractVector{T}) where T <: Real = zero(T)
Distributions.pdf(d::AbstractMvUrnDistribution, x::AbstractVector{<:Integer}) = exp(Distributions.logpdf(d, x))

# TODO: perhaps this should just be Int? Or create some other way to specify the type when doing rand?
Distributions.eltype(::AbstractMvUrnDistribution{T}) where T = T

in_eqsupport(d::AbstractMvUrnDistribution, no_parameters::T) where T<:Integer = one(T) <= no_parameters <= length(d)
function in_eqsupport(d::AbstractMvUrnDistribution, x::AbstractVector{T}) where T<:Integer
	length(x) == length(d) || return false
	for elem in x
		in_eqsupport(d, elem) || return false
	end
	return true
end
function Distributions._rand!(rng::Random.AbstractRNG, d::AbstractMvUrnDistribution, x::AbstractArray{T,2}) where T
	for i in axes(x, 2)
		Distributions._rand!(rng, d, view(x, :, i))
	end
	x
end

function Distributions._rand!(rng::Random.AbstractRNG, d::AbstractMvUrnDistribution, x::AbstractVector{T}) where {T<:Integer}

	probvec = zeros(Float64, length(x))

	for i in eachindex(x)

		_pdf_helper!(probvec, d, T(i), x)
		x[i] = rand(rng, Distributions.Categorical(probvec))

	end
	x
end

# TODO: rename no_parameters
function logpdf_incl(d::AbstractMvUrnDistribution, no_parameters::T) where T<:Integer
	in_eqsupport(d, no_parameters) || return T === BigInt ? BigFloat(-Inf) : Float64(-Inf)
	k = length(d)
	logpdf_model_distinct(d, no_parameters) + logstirlings2(k, no_parameters)
end
pdf_incl(d::AbstractMvUrnDistribution,  no_parameters) = exp(logpdf_incl(d,  no_parameters))



function logpdf_model(d::AbstractMvUrnDistribution, x::T) where T <: Integer
	in_eqsupport(d, x) || return T === BigInt ? BigFloat(-Inf) : Float64(-Inf)
	logpdf_model_distinct(d, x) - log_count_combinations(length(d), x)
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

logpdf_model_distinct(d::UniformMvUrnDistribution, ::AbstractVector{T}) where T <: Integer = -logbellnumr(convert(T, length(d)), zero(T))
logpdf_model_distinct(d::UniformMvUrnDistribution, ::T) where T <: Integer = -logbellnumr(convert(T, length(d)), zero(T))



#endregion

#region BetaBinomialMvUrnDistribution
struct BetaBinomialMvUrnDistribution{T <: Integer} <: AbstractMvUrnDistribution{T}
	k::T
	α::Float64
	β::Float64
	function BetaBinomialMvUrnDistribution(k::T, α::Float64 = 1.0, β::Float64 = 1.0) where T<:Integer
		new{T}(k, α, β)
	end
end

function BetaBinomialMvUrnDistribution(k::Integer, α::Number, β::Number = 1.0)
	BetaBinomialMvUrnDistribution(k, convert(Float64, α), convert(Float64, β))
end


# function Distributions.logpdf(d::BetaBinomialMvUrnDistribution, x::AbstractVector{<:Integer})
# 	log_model_probs_by_incl(d)[count_equalities(x) + 1]
# 	# Distributions.logpdf(Distributions.BetaBinomial(length(d) - 1, d.α, d.β), count_equalities(x))
# end

function log_model_probs_by_incl(d::BetaBinomialMvUrnDistribution{T}) where T
	Distributions.logpdf.(Distributions.BetaBinomial(d.k - one(T), d.α, d.β), zero(T):d.k - one(T)) .- log_expected_equality_counts(d.k)
end
function log_model_probs_by_incl(d::BetaBinomialMvUrnDistribution{T}, no_parameters::Integer) where T
	Distributions.logpdf(Distributions.BetaBinomial(d.k - one(T), d.α, d.β), no_parameters - one(T)) - log_expected_equality_counts(d.k, no_parameters)
end
logpdf_model_distinct(d::BetaBinomialMvUrnDistribution, x::AbstractVector{<:Integer}) = logpdf_model_distinct(d, count_parameters(x))
function logpdf_model_distinct(d::BetaBinomialMvUrnDistribution, no_parameters::Integer)
	in_eqsupport(d, no_parameters) || return -Inf
	# log_model_probs_by_incl(d)[no_parameters]
	log_model_probs_by_incl(d, no_parameters)
end

function logpdf_incl(d::BetaBinomialMvUrnDistribution, no_parameters::Integer)
	in_eqsupport(d, no_parameters) || return -Inf
	Distributions.logpdf(Distributions.BetaBinomial(length(d) - 1, d.α, d.β), no_parameters - 1)
end

#endregion

#region CustomInclusionMvUrnDistribution
"""
CustomInclusionMvUrnDistribution is similar to the BetaBinomialMvUrnDistribution in that the model probabilities are completely determined by the size of the partition.
Whereas the BetaBinomialMvUrnDistribution uses a BetaBinomial distribution to obtain the probabilities, the CustomInclusionMvUrnDistribution can be used to specify any vector of probabilities.
This distribution is particularly useful to sample uniformly from partitions of a given size.
For example:
```julia
rand(CustomInclusionMvUrnDistribution(4, ntuple(i->log(i==1), Val(4)))) # always all equal (1 parameter)
rand(CustomInclusionMvUrnDistribution(4, ntuple(i->log(i==3), Val(4)))) # always 3 parameters
rand(CustomInclusionMvUrnDistribution(4, ntuple(i->log(i==4), Val(4)))) # always completely distinct (4 parameters)
```
The function does not check if sum(exp, logpdf) ≈ 1.0, that is the callers responsibility.
"""
struct CustomInclusionMvUrnDistribution{T <: Integer, N} <: AbstractMvUrnDistribution{T}
	k::T
	logpdf::NTuple{N, Float64}
	function CustomInclusionMvUrnDistribution(k::T, logpdf::NTuple{N, Float64}) where {T<:Integer, N}
		k != length(logpdf) && throw(DomainError(logpdf, "Length musth match k"))
		new{T, N}(k, logpdf)
	end
end
CustomInclusionMvUrnDistribution(k::Integer, logpdf::AbstractVector) = CustomInclusionMvUrnDistribution(k, Tuple(logpdf))

log_model_probs_by_incl(d::CustomInclusionMvUrnDistribution) = d.logpdf .- log_expected_equality_counts(d.k)
log_model_probs_by_incl(d::CustomInclusionMvUrnDistribution, no_parameters::Integer) = d.logpdf[no_parameters] - log_expected_equality_counts(d.k)
logpdf_model_distinct(d::CustomInclusionMvUrnDistribution, x::AbstractVector{<:Integer}) = logpdf_model_distinct(d, count_parameters(x))
function logpdf_model_distinct(d::CustomInclusionMvUrnDistribution, no_parameters::Integer)
	in_eqsupport(d, no_parameters) || return -Inf
	log_model_probs_by_incl(d, no_parameters)
end

function logpdf_incl(d::CustomInclusionMvUrnDistribution, no_parameters::Integer)
	in_eqsupport(d, no_parameters) || return -Inf
	d.logpdf[no_parameters - 1]
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

function Distributions._rand!(rng::Random.AbstractRNG, d::RandomProcessMvUrnDistribution, x::AbstractVector{T}) where T<:Integer
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

function logpdf_incl(d::RandomProcessMvUrnDistribution, ::Integer)

	throw(DomainError(d, "only implemented for rpm<:Turing.RandomMeasures.DirichletProcess"))
	# generate one instance of all different models with no_parameters

	# result = zero(Float64)
	# opts = generate_distinct_models_with_sum(length(d), no_parameters)
	# for i in axes(opts, 2)
	# 	result += logpdf(d, view(opts, :, i)) + count_distinct_models_with_pattern(opts)
	# end
	# return result
end

function logpdf_model_distinct(d::RandomProcessMvUrnDistribution{W, T}, x::U) where {T<:Integer, U<:Integer, W}
	logpdf_model_distinct(d::RandomProcessMvUrnDistribution{W, T}, convert(T, x))
end

function logpdf_model_distinct(d::RandomProcessMvUrnDistribution{W, T}, no_parameters::T) where {T<:Integer, W}

	U = T === BigInt ? BigFloat : Float64
	in_eqsupport(d, no_parameters) || return U(-Inf)

	n = T(length(d))
	M = U(d.rpm.α)

	f! = x->filter!(y->length(y) == no_parameters, x)
	counts, sizes = count_set_partitions_given_partition_size(f!, n)

	res = zero(U)
	for i in eachindex(counts)#idx
		v = length(sizes[i]) * log(M) +
			SpecialFunctions.logabsgamma(M)[1] -
			SpecialFunctions.logabsgamma(M + n)[1] +
			sum(x->SpecialFunctions.logabsgamma(x)[1], sizes[i])

		res += counts[i] * v
	end
	return res / sum(counts)
end

# function logpdf_model_distinct(d::RandomProcessMvUrnDistribution{RPM, T}, urns::AbstractVector{<:Integer})  where {RPM<:RandomMeasures.DirichletProcess, T<:Integer}

# 	U = T === BigInt ? BigFloat : Float64
# 	in_eqsupport(d, urns) || return U(-Inf)

# 	n = length(d)
# 	M = d.rpm.α
# 	cc = StatsBase.countmap(urns)

# 	return length(cc) * log(M) +
# 		logabsgamma(M) -
# 		logabsgamma(M + n) +
# 		sum(logabsgamma, values(cc))

# end

function logpdf_model_distinct(d::RandomProcessMvUrnDistribution{RPM, T}, partition::AbstractVector{<:Integer})  where {RPM<:Turing.RandomMeasures.DirichletProcess, T<:Integer}

	U = T === BigInt ? BigFloat : Float64
	in_eqsupport(d, partition) || return U(-Inf)

	n = length(d)
	M = d.rpm.α
	cc = fast_countmap_partition(partition)

	return length(cc) * log(M) +
		EqualitySampler.logabsgamma(M) -
		EqualitySampler.logabsgamma(M + n) +
		sum(EqualitySampler.logabsgamma, cc)

end


function logpdf_incl(d::RandomProcessMvUrnDistribution{RPM, T}, no_parameters::Integer) where {RPM<:RandomMeasures.DirichletProcess, T<:Integer}

	#=
		From chapter 3: Dirichlet Process, equation 3.6
		Peter Müller, Abel Rodriguez
		NSF-CBMS Regional Conference Series in Probability and Statistics, 2013: 23-41 (2013) https://doi.org/10.1214/cbms/1362163748

		In 3.6 the extra n! term cancels when normalizing to a probability
	=#

	n = length(d)
	M = d.rpm.α
	k = no_parameters # number of unique values

	logunsignedstirlings1(n, k) +
		# SpecialFunctions.logfactorial(n) +
		k * log(M) +
		logabsgamma(M) -
		logabsgamma(M + n)

end

expected_inclusion_probabilities(d::RandomProcessMvUrnDistribution) = [pdf_incl(d, i) for i in 1:length(d)]
