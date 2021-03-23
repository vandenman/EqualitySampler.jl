using  Test
import Distributions, Turing

@testset "Multivariate urn distributions" begin

	noSamples = 10_000

	ks = 2:5
	αs = 0.5:0.5:2
	βs = 0.5:0.5:2

	αs2 = (1.0, 1.877, 2.0)

	Dset = (
		(
			dist = UniformMvUrnDistribution,
			args = ks
		),
		(
			dist = BetaBinomialMvUrnDistribution,
			args = Iterators.product((ks, αs, βs)...)
		),
		(
			dist = (k, α) -> RandomProcessMvUrnDistribution(k, Turing.RandomMeasures.DirichletProcess(α)),
			args = Iterators.product((ks, αs2)...)
		)
	)

	for dist in Dset

		@testset "Distribution $(dist[:dist])" begin

			for args in dist[:args]

				k = first(args)
				d = dist[:dist](args...)
				m = generate_distinct_models(k)

				@testset "pdf sums to 1 + inclusion probabilities match " begin

					s = [count_equalities(col) for col in eachcol(m)]
					indices = [findall(==(i), s) for i in 0:k-1]

					probs = [Distributions.pdf(d, col) for col in eachcol(m)]

					# pdf over all models sums to 1.0
					@test sum(probs) ≈ 1.0

					brute_force_incl_probs = [sum(probs[indices[i]]) for i in 1:k]
					efficient_incl_probs = [pdf_incl(d, i-1) for i in 1:k]

					# direct computation of inclusion probabilities (which is more efficient) equals brute force computation of inclusion probabilities
					@test efficient_incl_probs ≈ brute_force_incl_probs
				end

				@testset "simulated properties match theoretical results" begin


					samples = Matrix{Int}(undef, k, noSamples)
					for i in axes(samples, 2)
						v = view(samples, :, i)
						Distributions.rand!(d, v)
						v .= reduce_model(v)
					end


					if hasmethod(expected_model_probabilities, (RandomProcessMvUrnDistribution, ))
						empirical_model_probs     = collect(values(empirical_model_probabilities(samples)))
						expected_model_probs      = expected_model_probabilities(d)
					else # for the DirichletProcess this method doesn't exist (yet) so we brute force it
						tmp                       = empirical_model_probabilities(samples)
						empirical_model_probs     = collect(values(tmp))
						expected_model_probs      = [Distributions.pdf(d, reduce_model_dpp(parse.(Int, split(model, "")))) for model in keys(tmp)]
					end

					empirical_inclusion_probs = collect(values(empirical_inclusion_probabilities(samples)))
					expected_inclusion_probs  = expected_inclusion_probabilities(d)

					rtol = 0.15 + 0.02k # TODO: something better than this.
					@test isapprox(empirical_model_probs, expected_model_probs, rtol = rtol)
					@test isapprox(empirical_inclusion_probs, expected_inclusion_probs, rtol = rtol)

				end

				if hasmethod(log_expected_inclusion_probabilities, (RandomProcessMvUrnDistribution, ))
					@testset "Batch computations match individual ones" begin

						expected = log_expected_inclusion_probabilities(d)
						observed = logpdf_incl.(Ref(d), 0:length(d) - 1)

						@test isapprox(expected, observed)

					end
				end
			end
		end

	end
end
