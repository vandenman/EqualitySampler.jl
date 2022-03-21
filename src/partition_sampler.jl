
# just a hack for GibbsConditional

mutable struct PartitionSampler <: Distributions.DiscreteMultivariateDistribution
	size::Int
	nextValues::Vector{Int}
	logposterior::Function
	PartitionSampler(size::Int, logposterior::Function) = new(size, ones(Int, size), logposterior)
end

Base.rand(::Random.AbstractRNG, d::PartitionSampler) = d.nextValues

function (o::PartitionSampler)(c)
	o.nextValues = sample_next_values(c, o)
	return o
end

function sample_next_values(c, o)

	n_groups = length(c.partition)
	probvec = zeros(Float64, n_groups)
	nextValues = copy(c.partition)
	cache_idx = 0
	cache_value = -Inf # defined here to extend the scope beyond the if statement and for loop

	#=
		TODO: rather than enumerating all values, this could also enumerate the distinct models
		for example, given partition = [1, 2, 1, 1, 1] and j = 2 it makes no sense for to enumerate i = 1:5
		instead we can recognize that [1, 2, 1, 1, 1] = [1, 3, 1, 1, 1] = ... = [1, 5, 1, 1, 1]
		this implies we would do
		probvec[1] = logpdf(..., partition = [1, 1, 1, 1, 1])    (i = 1)
		probvec[2:5] .= logpdf(..., partition = [1, 2, 1, 1, 1]) (i = 2)
		which should save a bunch of likelihood evaluations whenever the current partition is far a model that implies everything is distinct
	=#

	new_label_log_posterior = 0.0
	new_label_log_posterior_computed = false
	present_labels = fast_countmap_partition_incl_zero(nextValues)

	# ~O(k^2) (double look over k) with at worst k * (k-1) likelihood evaluations if all labels are distinct and at best k * 2 likelihood evaluations
	for j in eachindex(probvec)

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

		probvec_normalized = exp.(probvec .- logsumexp_batch(probvec))
		if !Distributions.isprobvec(probvec_normalized)
			@show probvec, probvec_normalized
			@warn "probvec condition not satisfied! trying to normalize once more"
			probvec_normalized ./= sum(probvec_normalized)
		end

		if Distributions.isprobvec(probvec_normalized)
			# TODO: consider disabling the isprobvec check with check_args = false?
			present_labels[nextValues[j]] -= 1
			nextValues[j] = rand(Distributions.Categorical(probvec_normalized))
			present_labels[nextValues[j]] += 1

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

	# ~O(k^2) (double look over k) with k * (k-1) likelihood evaluations
	# for j in eachindex(probvec)

	# 	# originalValue = nextValues[j]
	# 	for i in eachindex(probvec)

	# 		# Maybe cache the one value that can be reused?
	# 		nextValues[j] = i
	# 		if nextValues[j] != cache_idx
	# 			probvec[i] = o.logposterior(nextValues, c)
	# 			# @show i, nextValues, probvec
	# 		else
	# 			# println("use cache")
	# 			# @show i cache_idx cache_value nextValues
	# 			probvec[i] = cache_value
	# 		end

	# 	end

	# 	probvec_normalized = exp.(probvec .- logsumexp_batch(probvec))
	# 	if !Distributions.isprobvec(probvec_normalized)
	# 		@show probvec, probvec_normalized
	# 		@warn "probvec condition not satisfied! trying to normalize once more"
	# 		probvec_normalized ./= sum(probvec_normalized)
	# 	end

	# 	if Distributions.isprobvec(probvec_normalized)
	# 		# TODO: consider disabling the isprobvec check with check_args = false?
	# 		nextValues[j] = rand(Distributions.Categorical(probvec_normalized))
	# 	elseif all(isinf, probvec)
	# 		return nextValues
	# 	else
	# 		nextValues[j] = c.partition[j]
	# 	end
	# 	if j != length(probvec)

	# 		cache_idx = nextValues[j+1]
	# 		cache_value = probvec[nextValues[j]]

	# 		# println("updated cache")
	# 		# @show cache_idx cache_value nextValues probvec
	# 	end
	# end
	# return nextValues
end