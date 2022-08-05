DynamicPPL.@model function proportion_model_full(successes, observations, partition = nothing, ::Type{T} = Float64) where T

	# Distributions.Beta() defaults to Distributions.Beta(1.0, 1.0)
	p_raw ~ DistributionsAD.filldist(Distributions.Beta(), length(successes))
	p_constrained = isnothing(partition) ? p_raw : average_equality_constraints(p_raw, partition)
	successes ~ Distributions.product_distribution(Distributions.Binomial.(observations, p_constrained))
	return p_constrained
end

DynamicPPL.@model function proportion_model_equality_selector(successes, observations, partition_prior)
	partition ~ partition_prior
	DynamicPPL.@submodel prefix="inner" p = proportion_model_full(successes, observations, partition)
	return p
end

function get_p_constrained(model, samps)

	default_result = DynamicPPL.generated_quantities(model, MCMCChains.get_sections(samps, :parameters))
	clean_result = Matrix{Float64}(undef, length(default_result[1][1]), size(default_result, 1))
	for i in eachindex(default_result)
		clean_result[:, i] .= default_result[i][1]
	end
	return vec(mean(clean_result, dims = 2)), clean_result
end

function get_proportion_model(successes, observations, partition_prior)

	if isnothing(partition_prior)
		model = proportion_model_full(successes, observations)
	else
		model = proportion_model_equality_selector(successes, observations, partition_prior)
	end

	return model
end

get_proportion_sampler(model, spl::Symbol, ϵ::Float64, n_leapfrog::Int) = get_sampler(model, spl, ϵ, n_leapfrog)
get_proportion_sampler(model, spl::Turing.Inference.InferenceAlgorithm, ::Float64, ::Int) = spl

"""
$(TYPEDSIGNATURES)

Fit independent binomials to the successes and observations.
If `partition_prior === nothing` then no constraints are imposed on the probabilities.
Otherwise, the model samples equalities among the proportions using the `partition_prior` as prior distribution.
"""
function proportion_test(
		successes::AbstractVector{T}, observations::AbstractVector{T},
		partition_prior::Union{Nothing, AbstractPartitionDistribution}
		;
		spl::Union{Symbol, Turing.Inference.InferenceAlgorithm} = :custom,
		mcmc_settings::MCMCSettings = MCMCSettings(),
		ϵ::Float64 = 0.05,
		n_leapfrog::Int = 20,
		kwargs...
	) where T<:Integer

	length(successes) != length(observations) && throw(ArgumentError("length(successes) != length(observations)"))
	if !isnothing(partition_prior)
		length(successes) != length(partition_prior) && throw(ArgumentError("length(successes) != length(partition_prior)"))
	end

	model = get_proportion_model(successes, observations, partition_prior)
	sampler = get_proportion_sampler(model, spl, ϵ, n_leapfrog)

	chain = sample_model(model, sampler, mcmc_settings; kwargs...);
	if isnothing(partition_prior)
		return chain
	else
		return combine_chain_with_generated_quantities(model, chain, "p_constrained")
	end

end