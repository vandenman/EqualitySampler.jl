#= NOTES

	*_distinct	does NOT account for duplicate configurations. For example, [1, 1, 1] is     the only null model and [2, 2, 2] does NOT exist
	*			does     account for duplicate configurations. For example, [1, 1, 1] is NOT the only null model and [2, 2, 2] does     exist (and so does [3, 3, 3])

=#



#region AbstractPartitionDistribution
"""
AbstractPartitionDistribution{<:Integer} <: Distributions.DiscreteMultivariateDistribution

Supertype for distributions over partitions.
"""
abstract type AbstractPartitionDistribution{T} <: Distributions.DiscreteMultivariateDistribution where T <: Integer end

Base.length(d::AbstractPartitionDistribution) = d.k
Distributions.minimum(::AbstractPartitionDistribution{T}) where T = one(T)
Distributions.maximum(d::AbstractPartitionDistribution{T}) where T = T(length(d))

Distributions.logpdf(::AbstractPartitionDistribution, ::AbstractVector{T}) where T <: Real = -Inf
Distributions.pdf(::AbstractPartitionDistribution, ::AbstractVector{T}) where T <: Real = zero(T)
Distributions.pdf(d::AbstractPartitionDistribution, x::AbstractVector{<:Integer}) = exp(Distributions.logpdf(d, x))

# TODO: perhaps this should just be Int? Or create some other way to specify the type when doing rand?
Distributions.eltype(::AbstractPartitionDistribution{T}) where T = T

in_eqsupport(d::AbstractPartitionDistribution, no_parameters::T) where T<:Integer = one(T) <= no_parameters <= length(d)
function in_eqsupport(d::AbstractPartitionDistribution, x::AbstractVector{T}) where T<:Integer
	length(x) == length(d) || return false
	for elem in x
		in_eqsupport(d, elem) || return false
	end
	return true
end
function Distributions._rand!(rng::Random.AbstractRNG, d::AbstractPartitionDistribution, x::AbstractArray{T,2}) where T
	for i in axes(x, 2)
		Distributions._rand!(rng, d, view(x, :, i))
	end
	x
end

function Distributions._rand!(rng::Random.AbstractRNG, d::AbstractPartitionDistribution, x::AbstractVector{T}) where {T<:Integer}

	probvec = zeros(Float64, length(x))

	for i in eachindex(x)

		_pdf_helper!(probvec, d, T(i), x)
		x[i] = rand(rng, Distributions.Categorical(probvec))

	end
	x
end

# TODO: rename no_parameters
"""
logpdf_incl(d::AbstractPartitionDistribution, no_parameters::Integers)

Log probability of all partitions with a particular number of parameters.
"""
function logpdf_incl(d::AbstractPartitionDistribution, no_parameters::T) where T<:Integer
	in_eqsupport(d, no_parameters) || return T === BigInt ? BigFloat(-Inf) : Float64(-Inf)
	k = length(d)
	logpdf_model_distinct(d, no_parameters) + logstirlings2(k, no_parameters)
end
"""
pdf_incl(d::AbstractPartitionDistribution, no_parameters::Integers)

Probability of all partitions with a particular number of parameters.
"""
pdf_incl(d::AbstractPartitionDistribution,  no_parameters) = exp(logpdf_incl(d,  no_parameters))


"""
logpdf_model(d::AbstractPartitionDistribution, x::Integer)
logpdf_model(d::AbstractPartitionDistribution, x::AbstractVector{<:Integer})

Synonym for `logpdf(d::AbstractPartitionDistribution, x)`, computes the log probability of a partition.
"""
function logpdf_model(d::AbstractPartitionDistribution, x::T) where T <: Integer
	in_eqsupport(d, x) || return T === BigInt ? BigFloat(-Inf) : Float64(-Inf)
	logpdf_model_distinct(d, x) - log_count_combinations(length(d), x)
end
logpdf_model(d::AbstractPartitionDistribution, x::AbstractVector{T}) where T <: Integer = logpdf_model_distinct(d, x) - log_count_combinations(x)

"""
pdf_model(d::AbstractPartitionDistribution, x::Integer)
pdf_model(d::AbstractPartitionDistribution, x::AbstractVector{<:Integer})

Synonym for `pdf(d::AbstractPartitionDistribution, x)`, computes the probability of a partition.
"""
pdf_model(d::AbstractPartitionDistribution, x) = exp(logpdf_model(d, x))
"""
pdf_model_distinct(d::AbstractPartitionDistribution, x)

Computes the probability of a partition without considering duplicated partitions (i.e., assuming all partitions are unique).
"""
pdf_model_distinct(d::AbstractPartitionDistribution, x) = exp(logpdf_model_distinct(d, x))

Distributions.logpdf(d::AbstractPartitionDistribution, x::AbstractVector{T}) where T<:Integer = logpdf_model(d, x)

#endregion

#region UniformPartitionDistribution
"""
UniformPartitionDistribution{T <: Integer} <: AbstractPartitionDistribution{T}

Uniform distribution over partitions.
"""
struct UniformPartitionDistribution{T <: Integer} <: AbstractPartitionDistribution{T}
	k::T
end



"""
logpdf_model_distinct(d::AbstractPartitionDistribution, x)

Computes the log probability of a partition without considering duplicated partitions (i.e., assuming all partitions are unique).
"""
logpdf_model_distinct(d::UniformPartitionDistribution, ::AbstractVector{T}) where T <: Integer = -logbellnumr(convert(T, length(d)), zero(T))
logpdf_model_distinct(d::UniformPartitionDistribution, ::T) where T <: Integer = -logbellnumr(convert(T, length(d)), zero(T))



#endregion

#region BetaBinomialPartitionDistribution
"""
BetaBinomialPartitionDistribution{T <: Integer} <: AbstractPartitionDistribution{T}

Beta binomial distribution over partitions.
If ``\\rho \\sim \\text{BetaBinomialPartitionDistribution}(k, \\alpha, \\beta)`` then ``\\text{count_parameters}(\\rho)\\sim \\text{BetaBinomial}(k - 1, \\alpha, \\beta)``.

"""
struct BetaBinomialPartitionDistribution{T <: Integer} <: AbstractPartitionDistribution{T}
	k::T
	α::Float64
	β::Float64
	function BetaBinomialPartitionDistribution(k::T, α::Float64 = 1.0, β::Float64 = 1.0) where T<:Integer
		new{T}(k, α, β)
	end
end

function BetaBinomialPartitionDistribution(k::Integer, α::Number, β::Number = 1.0)
	BetaBinomialPartitionDistribution(k, convert(Float64, α), convert(Float64, β))
end


function log_model_probs_by_incl(d::BetaBinomialPartitionDistribution{T}) where T
	Distributions.logpdf.(Distributions.BetaBinomial(d.k - one(T), d.α, d.β), zero(T):d.k - one(T)) .- log_expected_equality_counts(d.k)
end
function log_model_probs_by_incl(d::BetaBinomialPartitionDistribution{T}, no_parameters::Integer) where T
	Distributions.logpdf(Distributions.BetaBinomial(d.k - one(T), d.α, d.β), no_parameters - one(T)) - log_expected_equality_counts(d.k, no_parameters)
end
logpdf_model_distinct(d::BetaBinomialPartitionDistribution, x::AbstractVector{<:Integer}) = logpdf_model_distinct(d, count_parameters(x))
function logpdf_model_distinct(d::BetaBinomialPartitionDistribution, no_parameters::Integer)
	in_eqsupport(d, no_parameters) || return -Inf
	# log_model_probs_by_incl(d)[no_parameters]
	log_model_probs_by_incl(d, no_parameters)
end

function logpdf_incl(d::BetaBinomialPartitionDistribution, no_parameters::Integer)
	in_eqsupport(d, no_parameters) || return -Inf
	Distributions.logpdf(Distributions.BetaBinomial(length(d) - 1, d.α, d.β), no_parameters - 1)
end

#endregion

#region CustomInclusionPartitionDistribution
"""
CustomInclusionPartitionDistribution is similar to the BetaBinomialPartitionDistribution in that the model probabilities are completely determined by the size of the partition.
Whereas the BetaBinomialPartitionDistribution uses a BetaBinomial distribution to obtain the probabilities, the CustomInclusionPartitionDistribution can be used to specify any vector of probabilities.
This distribution is particularly useful to sample uniformly from partitions of a given size.
For example:
```julia
rand(CustomInclusionPartitionDistribution(4, ntuple(i->log(i==1), Val(4)))) # always all equal (1 parameter)
rand(CustomInclusionPartitionDistribution(4, ntuple(i->log(i==3), Val(4)))) # always 3 parameters
rand(CustomInclusionPartitionDistribution(4, ntuple(i->log(i==4), Val(4)))) # always completely distinct (4 parameters)
```
The function does not check if sum(exp, logpdf) ≈ 1.0, that is the callers responsibility.
"""
struct CustomInclusionPartitionDistribution{T <: Integer, N} <: AbstractPartitionDistribution{T}
	k::T
	logpdf::NTuple{N, Float64}
	function CustomInclusionPartitionDistribution(k::T, logpdf::NTuple{N, Float64}) where {T<:Integer, N}
		k != length(logpdf) && throw(DomainError(logpdf, "Length musth match k"))
		new{T, N}(k, logpdf)
	end
end
CustomInclusionPartitionDistribution(k::Integer, logpdf::AbstractVector) = CustomInclusionPartitionDistribution(k, Tuple(logpdf))

log_model_probs_by_incl(d::CustomInclusionPartitionDistribution) = d.logpdf .- log_expected_equality_counts(d.k)
log_model_probs_by_incl(d::CustomInclusionPartitionDistribution, no_parameters::Integer) = d.logpdf[no_parameters] - log_expected_equality_counts(d.k)
logpdf_model_distinct(d::CustomInclusionPartitionDistribution, x::AbstractVector{<:Integer}) = logpdf_model_distinct(d, count_parameters(x))
function logpdf_model_distinct(d::CustomInclusionPartitionDistribution, no_parameters::Integer)
	in_eqsupport(d, no_parameters) || return -Inf
	log_model_probs_by_incl(d, no_parameters)
end

function logpdf_incl(d::CustomInclusionPartitionDistribution, no_parameters::Integer)
	in_eqsupport(d, no_parameters) || return -Inf
	@inbounds d.logpdf[no_parameters - 1]
end


#endregion
"""
RandomProcessPartitionDistribution{RPM <: Turing.RandomMeasures.AbstractRandomProbabilityMeasure, T <: Integer} <: AbstractPartitionDistribution{T}

Distribution over partitions defined by a Random Probabiltiy Measure (RPM) as defined in Turing.RandomMeasures.
"""
struct RandomProcessPartitionDistribution{RPM <: Turing.RandomMeasures.AbstractRandomProbabilityMeasure, T <: Integer} <: AbstractPartitionDistribution{T}
	k::T
	rpm::RPM
end
rpm(d::RandomProcessPartitionDistribution) = d.rpm

"""
DirichletProcessPartitionDistribution(k::Integer, α::Float64)
DirichletProcessPartitionDistribution(k::Integer, ::Symbol = :Gopalan_Berry)
DirichletProcessPartitionDistribution(k::Integer, α::Real)

Wrapper function to create an object representing a Dirichlet process prior. These call RandomProcessPartitionDistribution but are a bit more user friendly.
Either set α directly by passing a float, or pass (any) symbol to use `EqualitySampler.dpp_find_α` to specify α, which uses the heuristic by
Gopalan & Berry (1998) so that P(everything equal) == P(everything unequal).
"""
DirichletProcessPartitionDistribution(k::Integer, α::Float64) = RandomProcessPartitionDistribution(k, Turing.RandomMeasures.DirichletProcess(α))
DirichletProcessPartitionDistribution(k::Integer, ::Symbol = :Gopalan_Berry) = DirichletProcessPartitionDistribution(k, dpp_find_α(k))
DirichletProcessPartitionDistribution(k::Integer, α::Real) = DirichletProcessPartitionDistribution(k, convert(Float64, α))
# whats the best way to do this ↓ ?
# const DirichletProcessPartitionDistributionType{T} = RandomProcessPartitionDistribution{Turing.RandomMeasures.DirichletProcess{Float64}, T}

function Distributions._rand!(rng::Random.AbstractRNG, d::RandomProcessPartitionDistribution, x::AbstractVector{T}) where T<:Integer
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

# Distributions.logpdf(::RandomProcessPartitionDistribution, ::AbstractVector{T}) where T<: Real = -Inf
# function Distributions.logpdf(d::RandomProcessPartitionDistribution, x::AbstractVector{T}) where T<: Integer

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

function logpdf_incl(d::RandomProcessPartitionDistribution, ::Integer)
	throw(DomainError(d, "only implemented for rpm<:Turing.RandomMeasures.DirichletProcess"))
end

function logpdf_model_distinct(d::RandomProcessPartitionDistribution{W, T}, x::U) where {T<:Integer, U<:Integer, W}
	logpdf_model_distinct(d::RandomProcessPartitionDistribution{W, T}, convert(T, x))
end

function logpdf_model_distinct(d::RandomProcessPartitionDistribution{W, T}, no_parameters::T) where {T<:Integer, W}

	U = T === BigInt ? BigFloat : Float64
	in_eqsupport(d, no_parameters) || return U(-Inf)

	n = T(length(d))
	M = U(d.rpm.α)

	f! = x->filter!(y->length(y) == no_parameters, x)
	counts, sizes = count_set_partitions_given_partition_size(f!, n)

	res = zero(U)
	for i in eachindex(counts)
		v = length(sizes[i]) * log(M) +
			SpecialFunctions.loggamma(M) -
			SpecialFunctions.loggamma(M + n) +
			sum(x->SpecialFunctions.loggamma(x), sizes[i])

		res += counts[i] * v
	end
	return res / sum(counts)
end

function logpdf_model_distinct(d::RandomProcessPartitionDistribution{RPM, T}, partition::AbstractVector{<:Integer})  where {RPM<:Turing.RandomMeasures.DirichletProcess, T<:Integer}

	U = T === BigInt ? BigFloat : Float64
	in_eqsupport(d, partition) || return U(-Inf)

	n = length(d)
	M = d.rpm.α
	cc = fast_countmap_partition(partition)

	return length(cc) * log(M) +
		SpecialFunctions.loggamma(M) -
		SpecialFunctions.loggamma(M + n) +
		sum(SpecialFunctions.loggamma, cc)

end


function logpdf_incl(d::RandomProcessPartitionDistribution{RPM, T}, no_parameters::Integer) where {RPM<:RandomMeasures.DirichletProcess, T<:Integer}

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
		SpecialFunctions.loggamma(M) -
		SpecialFunctions.loggamma(M + n)

end

expected_inclusion_probabilities(d::RandomProcessPartitionDistribution) = [pdf_incl(d, i) for i in 1:length(d)]
