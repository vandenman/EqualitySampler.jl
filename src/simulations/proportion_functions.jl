DynamicPPL.@model function proportion_model_full(no_errors, total_counts, partition = nothing, ::Type{T} = Float64) where T

	p_raw ~ DistributionsAD.filldist(Distributions.Beta(1.0, 1.0), length(no_errors))
	p_constrained = isnothing(partition) ? p_raw : average_equality_constraints(p_raw, partition)
	no_errors ~ Distributions.Product(Binomial.(total_counts, p_constrained))
	return (p_constrained, )
end

DynamicPPL.@model function proportion_model_equality_selector(no_errors, total_counts, partition_prior)
	partition ~ partition_prior
	DynamicPPL.@submodel prefix="inner" p = proportion_model_full(no_errors, total_counts, partition)
	return p
end

function get_p_constrained(model, samps)
	# TODO: delete this function?

	default_result = DynamicPPL.generated_quantities(model, MCMCChains.get_sections(samps, :parameters))
	clean_result = Matrix{Float64}(undef, length(default_result[1][1]), size(default_result, 1))
	for i in eachindex(default_result)
		clean_result[:, i] .= default_result[i][1]
	end
	return vec(mean(clean_result, dims = 2)), clean_result
end

function get_proportion_model_and_spl(no_errors, total_counts, partition_prior)

	if isnothing(partition_prior)
		model = proportion_model_full(no_errors, total_counts)
	else
		model = proportion_model_equality_selector(no_errors, total_counts, partition_prior)
	end

	return model
end

function proportion_test(
		no_success::AbstractVector{T}, total_observations::AbstractVector{T}, partition_prior::Union{Nothing, AbstractMvUrnDistribution};
		no_samples::Integer = 1_000, no_burnin::Integer = 500, no_chains::Integer = 3,
		spl = nothing,
		kwargs...
	) where T<:Integer

	model = get_proportion_model(no_success, total_observations, partition_prior)

	if isnothing(spl)
		spl = get_sampler(model)
	end

	# TODO: what package should be prefixed here?
	chain = sample(model, spl, no_samples, no_chains; discard_initial = no_burnin, kwargs...);
	return combine_chain_with_generated_quantities(model, chain, "p_constrained")


	# TODO: maybe don't return the model? also should this be a struct rather than a named tuple?
	# posterior_means, posterior_samples = get_p_constrained(model, all_samples)
	# return (posterior_means = posterior_means, posterior_samples = posterior_samples, all_samples = all_samples, model = model)

end