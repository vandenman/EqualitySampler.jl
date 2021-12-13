#=

	This file samples randomly from different distribution over partitions and then compares the results with
	the theoretical expressions. Sampling is first done with custom and afterward using Turing

	Distributions:

		UniformConditionalUrnDistribution
		BetaBinomialConditionalUrnDistribution
		DirichletProcessDistribution (Chinese restaurant)



=#

using EqualitySampler, Plots, Turing
include("simulations/samplePriorsTuring.jl")
include("simulations/plotFunctions.jl")

updateDistribution(::UniformConditionalUrnDistribution, urns, j) = UniformConditionalUrnDistribution(urns, j)
updateDistribution(D::BetaBinomialConditionalUrnDistribution, urns, j) = BetaBinomialConditionalUrnDistribution(urns, j, D.α, D.β)

function simulate_from_distribution(nrand, D)
	println("Drawing $nrand draws from $(typeof(D).name)")
	k = length(D)
	urns = copy(D.urns)
	sampled_models = Matrix{Int}(undef, k, nrand)
	for i in 1:nrand
		urns = ones(Int, k)
		for j in 1:k
			D = updateDistribution(D, urns, j)
			urns[j] = rand(D, 1)[1]
		end
		sampled_models[:, i] .= reduce_model(urns)
	end
	return sampled_models
end

k = 4
nrand = 100_000
urns = collect(1:k)
D = UniformConditionalUrnDistribution(urns, 1)
sampled_models = simulate_from_distribution(nrand, D)

empirical_model_probs     = empirical_model_probabilities(sampled_models)
empirical_inclusion_probs = empirical_equality_probabilities(sampled_models)

pjoint = visualize_eq_samples(D, empirical_model_probs, empirical_inclusion_probs)
# png(pjoint, "modelspace uniform $k.png")

nrand = 100_000
urns = collect(1:k)
D = BetaBinomialConditionalUrnDistribution(urns, 1, 1, 1)
sampled_models = simulate_from_distribution(nrand, D)

empirical_model_probs     = empirical_model_probabilities(sampled_models)
empirical_inclusion_probs = empirical_equality_probabilities(sampled_models)

pjoint = visualize_eq_samples(D, empirical_model_probs, empirical_inclusion_probs)

#region Turing UniformConditionalUrnDistribution
k = 4
uniform_reference_dist = UniformConditionalUrnDistribution(1:k)
equality_prior, empirical_model_probs, empirical_inclusion_probs, _ = sample_and_compute_Turing(k, uniform_prior = true)
visualize_eq_samples(equality_prior, empirical_model_probs, empirical_inclusion_probs)

equality_prior, empirical_model_probs, empirical_inclusion_probs, _ = sample_and_compute_Turing(k, uniform_prior = false)
visualize_eq_samples(equality_prior, empirical_model_probs, empirical_inclusion_probs)

# Note that this prior  puts different weight on 1221 than on 1222
empirical_model_probs, empirical_inclusion_probs, _ = sample_process_turing(k, Turing.RandomMeasures.DirichletProcess(1.0))
visualize_eq_samples(uniform_reference_dist, empirical_model_probs, empirical_inclusion_probs)

empirical_model_probs, empirical_inclusion_probs, _ = sample_process_turing(k, Turing.RandomMeasures.DirichletProcess(3.0))
visualize_eq_samples(uniform_reference_dist, empirical_model_probs, empirical_inclusion_probs)

empirical_model_probs, empirical_inclusion_probs, _ = sample_process_turing(k, Turing.RandomMeasures.PitmanYorProcess(0.5, 0.5, 1))
visualize_eq_samples(uniform_reference_dist, empirical_model_probs, empirical_inclusion_probs)
