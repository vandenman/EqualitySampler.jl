
# just a hack for GibbsConditional

struct PartitionSampler <: Distributions.DiscreteMultivariateDistribution
	size::Int
	nextValues::Vector{Int}
	logposterior::Function
	PartitionSampler(size::Int, logposterior::Function) = new(size, ones(Int, size), logposterior)
end

Base.rand(::Random.AbstractRNG, d::PartitionSampler) = d.nextValues

function (o::PartitionSampler)(c)
	o.nextValues .= sample_next_values(c, o)
	return o
end

function sample_next_values(c, o)

	n_groups = length(c.partition)
	probvec = zeros(Float64, n_groups)
	nextValues = copy(c.partition)
	cache_idx = 0
	cache_value = -Inf # defined here to extend the scope beyond the if statement and for loop

	new_label_log_posterior = 0.0
	new_label_log_posterior_computed = false
	present_labels = EqualitySampler.fast_countmap_partition_incl_zero(nextValues)

	# O(k^2) with at worst k*(k-1) likelihood evaluations if all labels are distinct and at best 2k likelihood evaluations
	@inbounds for j in eachindex(probvec)

		oldValue = nextValues[j]
		for i in eachindex(probvec)

			nextValues[j] = i
			if nextValues[j] == cache_idx

				probvec[i] = cache_value

			elseif !iszero(present_labels[i])

				probvec[i] = o.logposterior(nextValues, c)

			elseif new_label_log_posterior_computed

				probvec[i] = new_label_log_posterior

			else

				new_label_log_posterior = o.logposterior(nextValues, c)
				probvec[i] = new_label_log_posterior
				new_label_log_posterior_computed = true

			end

		end

		probvec_normalized = exp.(probvec .- LogExpFunctions.logsumexp(probvec))
		if !Distributions.isprobvec(probvec_normalized)
			@show probvec, probvec_normalized
			@warn "probvec condition not satisfied! trying to normalize once more"
			probvec_normalized ./= sum(probvec_normalized)
		end

		if Distributions.isprobvec(probvec_normalized)
			# decrement the occurence of the old value
			present_labels[oldValue] -= 1
			nextValues[j] = rand(Distributions.Categorical(probvec_normalized))
			# increment the occurence of the newly sampled value
			present_labels[nextValues[j]] += 1

			if !all(>=(0), present_labels) || sum(present_labels) != length(present_labels)
				@show present_labels
				error("This should be impossible!")
			end

		elseif all(isinf, probvec) # not possible to recover from this
			return nextValues
		else
			nextValues[j] = c.partition[j]
		end

		if j != length(probvec)
			cache_idx = nextValues[j+1]
			cache_value = probvec[nextValues[j]]
		end
		new_label_log_posterior_computed = false
	end
	return nextValues

end