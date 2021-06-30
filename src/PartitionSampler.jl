
# just a hack for GibbsConditional

mutable struct PartitionSampler <: Distributions.DiscreteMultivariateDistribution
	# counter::Int
	size::Int
	nextValues::Vector{Int}
	logposterior::Function
	# PartitionSampler(size::Int, logposterior::Function) = new(0, size, ones(Int, size), logposterior)
	PartitionSampler(size::Int, logposterior::Function) = new(size, ones(Int, size), logposterior)
end

Base.rand(::Random.AbstractRNG, d::PartitionSampler) = d.nextValues

function (o::PartitionSampler)(c)
	# @show c
	# o.counter += 1
	# if o.counter > length(c.partition)
	# 	o.counter = 0
	# end
	o.nextValues = sample_next_values(c, o)

	return o
end

function sample_next_values(c, o)

	n_groups = length(c.partition)
	probvec = zeros(Float64, n_groups)
	nextValues = copy(c.partition)
	cache_idx = 0
	cache_value = -Inf # defined here to extend the scope beyond the if statement and for loop

	for j in eachindex(probvec)
		# originalValue = nextValues[j]
		for i in eachindex(probvec)

			# Maybe cache the one value that can be reused?
			nextValues[j] = i
			if nextValues[j] != cache_idx
				probvec[i] = o.logposterior(nextValues, c)
			else
				# println("use cache")
				# @show i cache_idx cache_value nextValues
				probvec[i] = cache_value
			end

		end

		nextValues[j] = rand(Distributions.Categorical(exp.(probvec .- logsumexp_batch(probvec))))
		if j != length(probvec)

			cache_idx = nextValues[j+1]
			cache_value = probvec[nextValues[j]]

			# println("updated cache")
			# @show cache_idx cache_value nextValues probvec
		end
	end
	return nextValues
end