
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

	for j in eachindex(probvec)
		# originalValue = nextValues[j]
		for i in eachindex(probvec)

			nextValues[j] = i
			probvec[i] = o.logposterior(nextValues, c)

			# this should be a bit more general
			# θ_cs = average_equality_constraints(θ_s, equal_indices)
			# probvec[i] = sum(logpdf(NormalSuffStat(obs_var[j], c.μ_grand + θ_cs[j], σ, obs_n[j]), obs_mean[j]) for j in 1:n_groups)

		end
		# s = logsumexp_batch(probvec)
		probvec .-= logsumexp_batch(probvec)
		nextValues[j] = rand(Distributions.Categorical(exp.(probvec)))
	end
	return nextValues
end