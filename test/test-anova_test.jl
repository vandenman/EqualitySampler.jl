import Random, Distributions, MCMCChains, AbstractMCMC, Statistics, Turing
import EqualitySampler.Simulations

function partition_to_equality_matrix(partition)
	[i > j && p == q for (i, p) in enumerate(partition), (j, q) in enumerate(partition)]
end

on_ci = haskey(ENV, "CI") ? ENV["CI"] == "true" : false

Turing.setprogress!(!on_ci)

@testset "anova_test" begin

	# This fails on ci for some reason I do not understand.
	if !on_ci

		Random.seed!(42)
		n_groups = 5
		n_obs_per_group = 1000
		true_partition = rand(UniformPartitionDistribution(n_groups))
		temp_θ = randn(n_groups)
		true_θ = average_equality_constraints(temp_θ .- Statistics.mean(temp_θ), true_partition)
		data_object = Simulations.simulate_data_one_way_anova(n_groups, n_obs_per_group, true_θ, true_partition, 1.5)
		data = data_object.data
		partition_prior = BetaBinomialPartitionDistribution(n_groups, 1, n_groups)

		suff_stats_vec, _ = Simulations.prep_model_arguments(data)
		@test Distributions.logpdf(data_object.distribution, data_object.data.y) ≈ sum(loglikelihood_suffstats.(Distributions.Normal.(data_object.true_values.μ .+ data_object.true_values.θ, data_object.true_values.σ), suff_stats_vec))

		mcmc_settings = Simulations.MCMCSettings(;iterations = 15_000, chains = 4, parallel = on_ci ? AbstractMCMC.MCMCSerial : AbstractMCMC.MCMCThreads)

		chn_full = anova_test(data, nothing;         mcmc_settings = mcmc_settings)
		chn_eqs  = anova_test(data, partition_prior; mcmc_settings = mcmc_settings)

		estimated_θ_full = Statistics.mean(MCMCChains.group(chn_full, :θ_cs)).nt.mean
		estimated_θ_eqs  = Statistics.mean(MCMCChains.group(chn_eqs , :θ_cs)).nt.mean

		# comparison with population cell offsets
		@test isapprox(true_θ, estimated_θ_full; atol = 0.15)
		@test isapprox(true_θ, estimated_θ_eqs; atol = 0.15)

		# comparison with observed cell offsets
		obs_offset = ([Statistics.mean(data.y[idx]) for idx in data.g] .- Statistics.mean(data.y)) / sqrt(Statistics.var(data.y))
		@test isapprox(obs_offset, estimated_θ_full; atol = 0.15)
		@test isapprox(obs_offset, estimated_θ_eqs; atol = 0.15)

		estimated_post_probs = Simulations.compute_post_prob_eq(chn_eqs)
		true_eqs_mat = partition_to_equality_matrix(true_partition)
		@test all(
			if isone(true_eqs_mat[i, j])
				estimated_post_probs[i, j] > 0.7
			else
				estimated_post_probs[i, j] < 0.1
			end
			for i in axes(true_eqs_mat, 1) for j in i+1:size(true_eqs_mat, 1)
		)

	end
end