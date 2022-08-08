import Random, Distributions, MCMCChains, AbstractMCMC, Statistics, Turing
import EqualitySampler.Simulations

function partition_to_equality_matrix(partition)
	[i > j && p == q for (i, p) in enumerate(partition), (j, q) in enumerate(partition)]
end

on_ci = haskey(ENV, "CI") ? ENV["CI"] == "true" : false
Turing.setprogress!(!on_ci)

@testset "proportion_test" begin

	Random.seed!(42)
	n_groups = 5
	true_partition     = rand(UniformPartitionDistribution(n_groups))
	temp_probabilities = rand(n_groups)
	true_probabilities = average_equality_constraints(temp_probabilities, true_partition)
	observations    = rand(100:200, n_groups)
	successes = rand(Distributions.product_distribution(Distributions.Binomial.(observations, true_probabilities)))
	partition_prior = BetaBinomialPartitionDistribution(n_groups, 1, n_groups)

	mcmc_settings = Simulations.MCMCSettings(;iterations = 15_000, chains = 4, parallel = on_ci ? AbstractMCMC.MCMCSerial : AbstractMCMC.MCMCThreads)

	chn_full = proportion_test(successes, observations, nothing;         mcmc_settings = mcmc_settings)
	chn_eqs  = proportion_test(successes, observations, partition_prior; mcmc_settings = mcmc_settings)

	estimated_probabilities_full = Statistics.mean(chn_full).nt.mean
	estimated_probabilities_eqs = Statistics.mean(MCMCChains.group(chn_eqs, :p_constrained)).nt.mean

	# comparison with population proportions
	@test isapprox(true_probabilities, estimated_probabilities_full; atol = 0.1)
	@test isapprox(true_probabilities, estimated_probabilities_eqs; atol = 0.1)

	# comparison with observed proportions
	obs_proportions = successes ./ observations
	@test isapprox(obs_proportions, estimated_probabilities_full; atol = 0.1)
	@test isapprox(obs_proportions, estimated_probabilities_eqs; atol = 0.1)

	estimated_post_probs = Simulations.compute_post_prob_eq(chn_eqs)
	true_eqs_mat = partition_to_equality_matrix(true_partition)
	@test all(
		if isone(true_eqs_mat[i, j])
			estimated_post_probs[i, j] > 0.9
		else
			estimated_post_probs[i, j] < 0.1
		end
		for i in axes(true_eqs_mat, 1) for j in i+1:size(true_eqs_mat, 1)
	)

end
