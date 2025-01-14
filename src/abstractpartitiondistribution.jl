#region AbstractPartitionDistribution
"""
```
AbstractPartitionDistribution{<:Integer} <: Distributions.DiscreteMultivariateDistribution
```

Supertype for distributions over partitions.
"""
abstract type AbstractPartitionDistribution{T} <: Distributions.DiscreteMultivariateDistribution where T <: Integer end

"""
```
AbstractPartitionDistribution{<:Integer} <: Distributions.DiscreteMultivariateDistribution
```

Supertype for distributions over partitions that are determined by random processes, for example the Dirichlet Process and the Pitman-Yor process.
"""
abstract type AbstractProcessPartitionDistribution{T} <: AbstractPartitionDistribution{T} end


Base.length(d::AbstractPartitionDistribution) = d.k
Distributions.minimum(::AbstractPartitionDistribution{T}) where T = one(T)
Distributions.maximum(d::AbstractPartitionDistribution{T}) where T = T(length(d))



# TODO: perhaps this should just be Int? Or create some other way to specify the type when doing rand?
Distributions.eltype(::AbstractPartitionDistribution{T}) where T = T

function Distributions.insupport(d::AbstractPartitionDistribution, x::AbstractVector{T}) where T<:Integer
	length(x) == length(d) || return false
	for elem in x
		Distributions.insupport(d, elem) || return false
	end
	return true
end
Distributions.insupport(d::AbstractPartitionDistribution, no_parameters::T) where T<:Integer = one(T) <= no_parameters <= length(d)

# TODO: already defined in Distributions?
# Distributions.pdf(d::AbstractPartitionDistribution, x::AbstractVector{<:Integer}) = exp(Distributions.logpdf(d, x))

# TODO: what are these two good for?
# Distributions.logpdf(::AbstractPartitionDistribution, ::AbstractVector{T}) where T <: Real = -Inf
# Distributions.pdf(::AbstractPartitionDistribution, ::AbstractVector{T}) where T <: Real = zero(T)

# TODO: deprecate these in favor of Distributions.insupport !
# in_eqsupport(d::AbstractPartitionDistribution, no_parameters::T) where T<:Integer = one(T) <= no_parameters <= length(d)
# function in_eqsupport(d::AbstractPartitionDistribution, x::AbstractVector{T}) where T<:Integer
# 	length(x) == length(d) || return false
# 	for elem in x
# 		in_eqsupport(d, elem) || return false
# 	end
# 	return true
# end

struct PartitionSampler{D<:AbstractPartitionDistribution, T<:Number, U<:Integer} <: Distributions.Sampleable{Distributions.Multivariate,Distributions.Discrete}
	d::D
	probvec::Vector{T}
	partition_sizes::Vector{U}
end

Base.length(s::PartitionSampler) = length(s.d)

function Distributions.sampler(d::AbstractPartitionDistribution)
	k = length(d)
	T = typeof(k)
	U = float(T)
	PartitionSampler(d, Vector{U}(undef, k), Vector{T}(undef, k))
end

"""
Fill `x` with 1 / length(x).
Special implementation because the naive fill!(x, 1/length(x)) is numerically inaccurate for BigFloat,
e.g., `fill!(x, 1/length(x))` is not the same as `fill!(x, inv(BigFloat(length(x))))`, only the latter sums to 1.
"""
function fill_uniform!(x)
	fill!(x, inv(convert(eltype(x), length(x))))
	return x
end

_pdf_helper!(s::PartitionSampler, i::Integer, x::AbstractVector{<:Integer}) = _pdf_helper!(s.probvec, s.d, i, x, s.partition_sizes)

function Distributions._rand!(rng::Random.AbstractRNG, d::Union{AbstractPartitionDistribution, PartitionSampler}, x::AbstractVector{T}) where {T<:Integer}

	s = Distributions.sampler(d)
	for i in eachindex(x)

		probvec = _pdf_helper!(s, T(i), x)
		x[i] = rand(rng, Distributions.Categorical(probvec))

	end
    maybe_reduce_model!(x, d)
	x
end

function maybe_reduce_model!(x::AbstractVector{<:Integer}, ::AbstractPartitionDistribution)
    reduce_model_2!(x)
end
function maybe_reduce_model!(x::AbstractVector{<:Integer}, d::PartitionSampler{})
    lookup = d.partition_sizes
    fill!(lookup, zero(eltype(lookup)))
    reduce_model_2!(x, lookup)
end


# TODO: rename no_parameters
# TODO: the method below is only valid for AbstractSizePartitionDistribution & uniform
"""
```
logpdf_incl(d::AbstractPartitionDistribution, no_parameters::Integers)
```

Log probability of the sum of all partitions with that number of parameters.
"""
# function logpdf_incl(d::AbstractPartitionDistribution, no_parameters::T) where T<:Integer
# 	Distributions.insupport(d, no_parameters) || return convert(float(T), -Inf)
# 	logpdf_model_distinct(d, no_parameters) + logstirlings2(length(d), no_parameters)
# end

"""
```
pdf_incl(d::AbstractPartitionDistribution, no_parameters::Integers)
```

Probability of the sum of all partitions with that number of parameters.
"""
pdf_incl(d::AbstractPartitionDistribution,  no_parameters) = exp(logpdf_incl(d,  no_parameters))



log_count_combinations(d::AbstractPartitionDistribution, x) = false


"""
```
logpdf_model(d::AbstractPartitionDistribution, x::Integer)
logpdf_model(d::AbstractPartitionDistribution, x::AbstractVector{<:Integer})
```

Synonym for `logpdf(d::AbstractPartitionDistribution, x)`, computes the log probability of a partition.
"""
function logpdf_model(d::AbstractPartitionDistribution, x::T) where T <: Integer
	Distributions.insupport(d, x) || return convert(float(T), -Inf)
	logpdf_model_distinct(d, x) - log_count_combinations(d, x)
end
logpdf_model(d::AbstractPartitionDistribution, x::AbstractVector{T}) where T <: Integer = logpdf_model_distinct(d, x) - log_count_combinations(d, x)

"""
```
pdf_model(d::AbstractPartitionDistribution, x::Integer)
pdf_model(d::AbstractPartitionDistribution, x::AbstractVector{<:Integer})
```

Synonym for `pdf(d::AbstractPartitionDistribution, x)`, computes the probability of a partition.
"""
pdf_model(d::AbstractPartitionDistribution, x) = exp(logpdf_model(d, x))
"""
```
pdf_model_distinct(d::AbstractPartitionDistribution, x)
```

Computes the probability of a partition without considering duplicated partitions (i.e., assuming all partitions are unique).
"""
pdf_model_distinct(d::AbstractPartitionDistribution, x) = exp(logpdf_model_distinct(d, x))

Distributions.logpdf(d::AbstractPartitionDistribution, x::AbstractVector{T}) where T<:Integer = logpdf_model(d, x)

# TODO: these are probably not a good idea, methods should define
# the probabilities directly?
function expected_inclusion_probabilities(d::AbstractPartitionDistribution)
	counts = expected_inclusion_counts(d)
	return counts ./ sum(counts)
end

function log_expected_inclusion_probabilities(d::AbstractPartitionDistribution)
	log_counts = log_expected_equality_counts(d)
	return log_counts .- LogExpFunctions.logsumexp(log_counts)
end


"""
```
prediction_rule(d::AbstractPartitionDistribution, r::Integer)
prediction_rule(d::Type{<:AbstractPartitionDistribution}, r::Integer, args...)
```

Compute the prediction rule for a partition distribution, i.e., the probability that a new observations belongs to a new (unseen) cluster.
The current number of clusters is given by `r`.

"""
function prediction_rule(d::AbstractPartitionDistribution, r::Integer)

    k = length(d)
    one(r) <= r <= k || throw(DomainError("r must be between 1 and $k"))
    partition = ones(typeof(k), k)
    partition[k-r+1:k-1] .= 2:r
    probvec = zeros(k)
    EqualitySampler._pdf_helper!(probvec, d, k, partition, zeros(Int, k))
	# NOTE: could avoid allocating an intermediate array here
    return sum(probvec[eachindex(probvec) .∉ Ref(1:r)])

end

prediction_rule(d::Type{<:AbstractPartitionDistribution}, r::Integer, args...) = prediction_rule(d(args...), r)

"""
```
tie_probability(d::AbstractPartitionDistribution)
```

Compute the probability that `i == j` in a given partition.
Assumes that `d` is an exchangeable partition distribution.
"""
function tie_probability(d::AbstractPartitionDistribution)
    probvec = zeros(Float64, length(d))
    x = zeros(Int, length(d))
    x[1] = 1
    partition_size = similar(x)
    EqualitySampler._pdf_helper!(probvec, d, 2, x, partition_size)
    return probvec[1]
end


#endregion
