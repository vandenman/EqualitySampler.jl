using  Test
import Distributions, Turing, StatsBase, Statistics
#=

	TODO:

	separate this into two parts:
		- tests on the properties of the distinct model space
		- tests on the properties of the nondistinct model space

=#

@testset "Multivariate urn distributions" begin

	noSamples = 15_000

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
				# m0 = generate_distinct_models(k)
				m = generate_all_models(k)

				@testset "args: $args" begin

					@testset "pdf sums to 1, inclusion probabilities match, model probabilities match " begin

						s = vec([count_equalities(collect(col)) for col in m])
						indices = [findall(==(i), s) for i in 0:k-1]

						probs = vec([Distributions.pdf(d, collect(col)) for col in m])
						# pdf over all models sums to 1.0
						@test sum(probs) ≈ 1.0

						brute_force_incl_probs = [sum(probs[indices[i]]) for i in 1:k]
						efficient_incl_probs = [pdf_incl(d, i-1) for i in 1:k]

						# direct computation of inclusion probabilities (which is more efficient) equals brute force computation of inclusion probabilities
						@test efficient_incl_probs ≈ brute_force_incl_probs

						# no. equalities is insufficient for model probabilities for DPP
						if !(d isa RandomProcessMvUrnDistribution)
							model_probs  = pdf_model.(Ref(d), 0:k-1)
							model_counts = count_distinct_models_with_incl.(k, 0:k-1) .* count_combinations.(k, k .- (0:k-1))

							# dividing inclusion probabilities by model size frequency gives the model probabilities
							@test efficient_incl_probs ./ model_counts ≈ model_probs
						end
					end

					@testset "simulated properties match theoretical results" begin

						# TODO: this test should NOT use reduce_model but just call rand!
						# samples = Distributions.rand(d, noSamples)

						samples = Matrix{Int}(undef, k, noSamples)
						for i in axes(samples, 2)
							v = view(samples, :, i)
							Distributions.rand!(d, v)
							v .= reduce_model(v)
						end

						if !(d isa RandomProcessMvUrnDistribution)
							empirical_model_probs     = collect(values(empirical_model_probabilities(samples)))
							expected_model_probs      = expected_model_probabilities(d)
						else # for the DirichletProcess this method doesn't exist (yet) so we brute force it
							tmp                       = empirical_model_probabilities(samples)
							empirical_model_probs     = collect(values(tmp))
							expected_model_probs      = [exp(EqualitySampler.logpdf_model_distinct(d, reduce_model_dpp(parse.(Int, split(model, ""))))) for model in keys(tmp)]
						end

						empirical_inclusion_probs = collect(values(empirical_inclusion_probabilities(samples)))
						expected_inclusion_probs  = expected_inclusion_probabilities(d)

						rtol = 0.15 + 0.02k # TODO: something better than this.
						@test isapprox(empirical_model_probs, expected_model_probs, rtol = rtol)
						@test isapprox(empirical_inclusion_probs, expected_inclusion_probs, rtol = rtol)

					end

					if !(d isa RandomProcessMvUrnDistribution)#hasmethod(log_expected_inclusion_probabilities, (RandomProcessMvUrnDistribution, ))
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

	@testset "logpdf_model_distinct works with DPP" begin

		for k in 2:10
			d = DirichletProcessMvUrnDistribution(k, .5)
			m = generate_distinct_models(k)

			lpdfs = mapslices(x->logpdf_model_distinct(d, x), m, dims = 1)
			manual_sizes = [sort!(collect(values(StatsBase.countmap(x))); rev = true) for x in eachcol(m)]
			counts, sizes = count_set_partitions_given_partition_size(k)

			for i in eachindex(sizes)

				idx = findall(==(sizes[i]), manual_sizes)

				@test length(idx) == counts[i]
				@test sum(lpdfs[idx]) ≈ counts[i] * lpdfs[idx[1]]

			end

			eqs = [count_equalities(x) for x in eachcol(m)]
			ueqs = unique(eqs)
			expected = [Statistics.mean(lpdfs[ueqs_i .== eqs]) for ueqs_i in ueqs]
			computed = logpdf_model_distinct.(Ref(d), ueqs)
			@test expected ≈ computed

		end
	end

	@testset "logpdf_model_distinct is equal for value and models" begin

		for k in 3:6

			d_u  = UniformMvUrnDistribution(k)
			d_bb = BetaBinomialMvUrnDistribution(k, k, 1)
			models = generate_distinct_models(k)
			equalities = count_equalities.(eachcol(models))

			logprob_d_bb_models     = logpdf_model_distinct.(Ref(d_bb), eachcol(models))
			logprob_d_bb_equalities = logpdf_model_distinct.(Ref(d_bb), equalities)

			logprob_d_u_models     = logpdf_model_distinct.(Ref(d_u), eachcol(models))
			logprob_d_u_equalities = logpdf_model_distinct.(Ref(d_u), equalities)

			@test logprob_d_bb_models ≈ logprob_d_bb_equalities
			@test logprob_d_u_models ≈ logprob_d_u_equalities

		end
	end
end


