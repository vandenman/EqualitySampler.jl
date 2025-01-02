
abstract type AbstractProcessPartitionDistribution{T} <: AbstractPartitionDistribution{T} end

function logpdf_model_distinct(::AbstractProcessPartitionDistribution, ::Integer)
	throw(ArgumentError("logpdf_model_distinct was called for an AbstractProcessPartitionDistribution (e.g., Dirichlet process, Pitman-Yor), with an integer as the second argument. However, this cannot be computed for these types of distributions."))
end
