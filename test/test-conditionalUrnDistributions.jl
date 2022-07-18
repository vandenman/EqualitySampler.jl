updateDistribution(::EqualitySampler.UniformConditionalUrnDistribution, urns, j) = EqualitySampler.UniformConditionalUrnDistribution(urns, j)
updateDistribution(D::EqualitySampler.BetaBinomialConditionalUrnDistribution, urns, j) = EqualitySampler.BetaBinomialConditionalUrnDistribution(urns, j, D.α, D.β)

@testset "Simulated Properties Match Theoretical Ones" begin

	Dset = [EqualitySampler.UniformConditionalUrnDistribution([1, 1], 1), EqualitySampler.BetaBinomialConditionalUrnDistribution([1, 1], 1, 1, 1)]
	ks = 2:5
	noSamples = 10_000
	for D in Dset
		for k in ks
			samples = Matrix{Int}(undef, k, noSamples)
			for i in 1:noSamples
				urns = ones(Int, k)
				for j in 1:k
					D = updateDistribution(D, urns, j)
					urns[j] = rand(D)
				end
				samples[:, i] .= EqualitySampler.reduce_model(urns)
			end
			D = updateDistribution(D, ones(Int, k), 1)
			empirical_model_probs     = collect(values(EqualitySampler.empirical_model_probabilities(samples)))
			empirical_inclusion_probs = collect(values(EqualitySampler.empirical_no_parameters_probabilities(samples)))
			expected_model_probs      = EqualitySampler.expected_model_probabilities(D)
			expected_inclusion_probs  = EqualitySampler.expected_inclusion_probabilities(D)
			rtol = 0.15 + 0.02k # TODO: something better than this.
			@testset "Distribution: $D" begin
				# correlations fail for uniform model because the theoretical values have 0 variance.
				# @test cor(empirical_model_probs, expected_model_probs) > 0.95
				# @test cor(empirical_inclusion_probs, expected_inclusion_probs) > 0.95
				@test isapprox(empirical_model_probs, expected_model_probs, rtol = rtol)
				@test isapprox(empirical_inclusion_probs, expected_inclusion_probs, rtol = rtol)
			end
		end
	end
end

