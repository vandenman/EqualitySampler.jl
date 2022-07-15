using Test
import Turing

@testset "test find_dpp_α" begin

	ks = 2:10

	function evaluate_α(α, k)

		d_test = RandomProcessMvUrnDistribution(k, Turing.RandomMeasures.DirichletProcess(α))

		null_model = fill(1, k)
		full_model = collect(1:k)

		abs(EqualitySampler.logpdf_model_distinct(d_test, full_model) - EqualitySampler.logpdf_model_distinct(d_test, null_model))

	end

	for k in ks

		α = @inferred EqualitySampler.dpp_find_α(k)
		@test isapprox(evaluate_α(α, k), 0.0; atol = 0.01)

	end
end