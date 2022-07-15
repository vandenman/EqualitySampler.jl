using  Test
import Distributions, Turing, StatsBase, Statistics
#=

	TODO:

	separate this into two parts:
		- tests on the properties of the distinct model space
		- tests on the properties of the nondistinct model space

=#

function reduce_model_dpp(x::AbstractVector{<:Integer})

	y = similar(x)
	currentMax = 0
	visited = Set{Int}()
	for i in eachindex(y)
		if x[i] ∉ visited
			currentMax += 1
			y[i] = currentMax
			for j in i+1:length(x)
				if x[i] == x[j]
					y[j] = currentMax
				end
			end
			push!(visited, x[i])
		end
	end
	return y
end

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

			for args in dist.args

				k = first(args)
				d = dist.dist(args...)
				# m0 = generate_distinct_models(k)
				m = generate_all_models(k)

				@testset "args: $args" begin

					@testset "pdf sums to 1, inclusion probabilities match, model probabilities match " begin

						s = vec([count_parameters(collect(col)) for col in m])
						indices = [findall(==(i), s) for i in 1:k]

						probs = vec([Distributions.pdf(d, collect(col)) for col in m])
						# pdf over all models sums to 1.0
						@test sum(probs) ≈ 1.0

						brute_force_incl_probs = [sum(probs[indices[i]]) for i in 1:k]
						efficient_incl_probs = [pdf_incl(d, i) for i in 1:k]

						# direct computation of inclusion probabilities (which is more efficient) equals brute force computation of inclusion probabilities
						@test efficient_incl_probs ≈ brute_force_incl_probs

						# no. equalities is insufficient for model probabilities for DPP
						if !(d isa RandomProcessMvUrnDistribution)
							model_probs  = pdf_model.(Ref(d), 1:k)
							model_counts = stirlings2.(k, 1:k) .* EqualitySampler.count_combinations.(k, 1:k)

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

							empirical_model_probs     = collect(values(EqualitySampler.empirical_model_probabilities(samples)))
							expected_model_probs      = EqualitySampler.expected_model_probabilities(d)
						else # for the DirichletProcess this method doesn't exist (yet) so we brute force it
							tmp                       = EqualitySampler.empirical_model_probabilities(samples)
							empirical_model_probs     = collect(values(tmp))
							expected_model_probs      = [exp(EqualitySampler.logpdf_model_distinct(d, reduce_model_dpp(parse.(Int, split(model, ""))))) for model in keys(tmp)]
						end

						empirical_inclusion_probs = collect(values(EqualitySampler.empirical_no_parameters_probabilities(samples)))
						expected_inclusion_probs  = EqualitySampler.expected_inclusion_probabilities(d)

						rtol = 0.15 + 0.02k # TODO: something better than this.
						@test isapprox(empirical_model_probs, expected_model_probs, rtol = rtol)
						@test isapprox(empirical_inclusion_probs, expected_inclusion_probs, rtol = rtol)

					end

					if !(d isa RandomProcessMvUrnDistribution)
						@testset "Batch computations match individual ones" begin

							expected = EqualitySampler.log_expected_inclusion_probabilities(d)
							observed = logpdf_incl.(Ref(d), 1:length(d))

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
			counts, sizes = EqualitySampler.count_set_partitions_given_partition_size(k)

			for i in eachindex(sizes)

				idx = findall(==(sizes[i]), manual_sizes)

				@test length(idx) == counts[i]
				@test sum(lpdfs[idx]) ≈ counts[i] * lpdfs[idx[1]]

			end

			parameter_counts = [count_parameters(x) for x in eachcol(m)]
			unique_parameter_counts = unique(parameter_counts)
			expected = [Statistics.mean(lpdfs[unique_parameter_counts_i .== parameter_counts]) for unique_parameter_counts_i in unique_parameter_counts]
			computed = logpdf_model_distinct.(Ref(d), unique_parameter_counts)
			@test expected ≈ computed

		end
	end

	@testset "logpdf_model_distinct is equal for value and models" begin

		for k in 3:6

			d_u  = UniformMvUrnDistribution(k)
			d_bb = BetaBinomialMvUrnDistribution(k, k, 1)
			models = generate_distinct_models(k)
			parameters = count_parameters.(eachcol(models))

			logprob_d_bb_models     = logpdf_model_distinct.(Ref(d_bb), eachcol(models))
			logprob_d_bb_parameters = logpdf_model_distinct.(Ref(d_bb), parameters)

			logprob_d_u_models     = logpdf_model_distinct.(Ref(d_u), eachcol(models))
			logprob_d_u_parameters = logpdf_model_distinct.(Ref(d_u), parameters)

			@test logprob_d_bb_models ≈ logprob_d_bb_parameters
			@test logprob_d_u_models ≈ logprob_d_u_parameters

		end
	end
end


