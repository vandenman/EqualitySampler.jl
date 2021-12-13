using Turing, Turing.RandomMeasures, Plots
import StatsBase: countmap

function sample_from_process(k::Int, rpm::RandomMeasures.AbstractRandomProbabilityMeasure; no_samples::Int = 100)

	result = Matrix{Int}(undef, k, no_samples)
	_sample_process!(result, rpm)
	result
end

function _sample_process!(result::AbstractMatrix, rpm::RandomMeasures.AbstractRandomProbabilityMeasure)
	for i in axes(result, 2)
		sample_process!(view(result, :, i), rpm)
	end
end

function _sample_process!(z::AbstractVector, rpm::RandomMeasures.AbstractRandomProbabilityMeasure)

	z[1] = 1
	for i in 2:length(z)
		# Number of observations per cluster.
		K = maximum(view(z, 1:i-1))
		nk = Vector{Int}(map(k -> sum(z .== k), 1:K))

		# Draw new assignment.
		z[i] = rand(ChineseRestaurantProcess(rpm, nk))
	end
end

function _sample_process!(z::AbstractVector, rpm::RandomMeasures.PitmanYorProcess)

	z[1] = 1
	for i in 2:length(z)
		# Number of observations per cluster.
		K = maximum(view(z, 1:i-1))
		nk = Vector{Int}(map(k -> sum(z .== k), 1:K))

		# Draw new assignment.
		_rpm = PitmanYorProcess(rpm.d, rpm.θ, sum(!iszero, nk))
		z[i] = rand(ChineseRestaurantProcess(_rpm, nk))
	end
end

@model TuringDirichletProcess(k, α) = begin

	rpm = DirichletProcess(α)
	z = tzeros(Int, k)

	for i in 1:k

		# Number of clusters.
		K = maximum(z)
		nk = Vector{Int}(map(k -> sum(z .== k), 1:K))

		# Draw the latent assignment.
		# z[i] ~ StickBreakingProcess(rpm)
		z[i] ~ ChineseRestaurantProcess(rpm, nk)

	end
end

@model TuringPitmanYorProcess(k, d, θ) = begin

	# Latent assignment.
	z = tzeros(Int, k)
	z[1] ~ Categorical(1)

	for i in 2:k

		# Number of clusters.
		K = maximum(z)
		nk = Vector{Int}(map(k -> sum(z .== k), 1:K))

		rpm = PitmanYorProcess(d, θ, sum(!iszero, nk))
		# Draw the latent assignment.
		z[i] ~ ChineseRestaurantProcess(rpm, nk)

	end
end

sample_process_turing(k::Int, rpm::RandomMeasures.DirichletProcess; no_samples::Int = 100_000) = _sample_process_turing(TuringDirichletProcess(k, rpm.α),        no_samples = no_samples)
sample_process_turing(k::Int, rpm::RandomMeasures.PitmanYorProcess; no_samples::Int = 100_000) = _sample_process_turing(TuringPitmanYorProcess(k, rpm.d, rpm.θ), no_samples = no_samples)

function _sample_process_turing(model; no_samples::Int = 100_000)

	samples = sample(model, Prior(), no_samples)
	nc = size(samples)[2]
	sampled_models = Matrix{Int}(undef, nc - 1, no_samples)
	for i in 1:no_samples
		sampled_models[:, i] .= reduce_model(Int.(samples.value.data[i, 2:nc]))
	end
	empirical_model_probs     = empirical_model_probabilities(sampled_models)
	empirical_inclusion_probs = empirical_equality_probabilities(sampled_models)
	return empirical_model_probs, empirical_inclusion_probs, sampled_models

end

@model function small_model(k, uniform = true, α = 1.0, β = 1.0)
	equal_indices = TArray{Int}(k)
	equal_indices .= 1 # we could drop this if we adjust the constructor of UniformConditionalUrnDistribution and friends
	for i in 1:k
		if uniform
			equal_indices[i] ~ UniformConditionalUrnDistribution(equal_indices, i)
		else
			equal_indices[i] ~ BetaBinomialConditionalUrnDistribution(equal_indices, i, α, β)
		end
	end
end

function sample_and_compute_Turing(k::Int; no_samples::Int = 100_000, uniform_prior::Bool, α::Float64 = 1.0, β::Float64 = 1.0)
	model = small_model(k, uniform_prior, α, β)
	samples = sample(model, Prior(), no_samples)
	empirical_model_probs = compute_model_probs(samples)
	empirical_inclusion_probs = compute_incl_probs(samples)
	D = uniform_prior ? D = UniformConditionalUrnDistribution(1:k, 1) : BetaBinomialConditionalUrnDistribution(1:k, 1, α, β)
	return D, empirical_model_probs, empirical_inclusion_probs, samples
end


# k = 4
# nsamples = 100_000
# rpm = DirichletProcess(1.0)
# samps = zeros(Int, k, nsamples)

# for j in 1:nsamples

# 	z = view(samps, :, j)

# 	for i in 1:k

# 		# Number of clusters.
# 		K = maximum(z)
# 		nk = Vector{Int}(map(k -> sum(z .== k), 1:K))
# 		z[i] = rand(ChineseRestaurantProcess(rpm, nk))

# 	end
# 	z .= reduce_model(z)
# end

# ref = UniformConditionalUrnDistribution(ones(Int, k), 1)

# empirical_model_probs     = empirical_model_probabilities(samps)
# empirical_inclusion_probs = empirical_equality_probabilities(samps)

# pjoint = visualize_eq_samples(ref, empirical_model_probs, empirical_inclusion_probs)
