using EqualitySampler, Test
import Distributions, StatsBase, Statistics
#=

    TODO:

    - would be nice if it runs a little bit faster?

=#

# TODO: no longer needed?
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

    no_samples = 15_000

    ks = 2:5
    αs = 0.5:0.5:2
    βs = 0.5:0.5:2

    αs2 = (0.1, 0.5, 0.9)
    θs = (0.1, 0.2, 1.4)


    log_probs = [
        log.(rand(Distributions.Dirichlet(ones(k) ./ k)))
        for k in ks
    ]
    log_counts = EqualitySampler.log_expected_equality_counts.(ks)

    Dset = (
        (
            dist = UniformPartitionDistribution,
            args = ks
        ),
        (
            dist = BetaBinomialPartitionDistribution,
            args = Iterators.product((ks, αs, βs)...)
        ),
        (
            dist = CustomInclusionPartitionDistribution,
            args = ks
        ),
        (
            dist = PrecomputedCustomInclusionPartitionDistribution,
            args = ks
        ),
        (
            dist = DirichletProcessPartitionDistribution,
            args = Iterators.product((ks, αs2)...)
        ),
        (
            dist = PitmanYorProcessPartitionDistribution,
            args = Iterators.product((ks, αs2, θs)...)
        )
    )

    dist, args0 = Dset[3]
    @testset "Distribution $dist" for (dist, args0) in Dset

        # dist = last(Dset)
        # args = first(dist.args)
        # args = (2, 1.877, 0.2)#first(Iterators.take(dist.args, 10))
        # @testset "Distribution $args" for args in dist.args

        @testset "args: $args" for args in args0

            k = first(args)
            d = if dist <: CustomInclusionPartitionDistribution
                d = dist(k, log_probs[k - 1])
            elseif dist <: PrecomputedCustomInclusionPartitionDistribution
                d = dist(k, log_probs[k - 1], log_counts[k - 1])
            else
                d = dist(args...)
            end

            m = PartitionSpace(k)#, EqualitySampler.DuplicatedPartitionSpace)

            @testset "pdf sums to 1, inclusion probabilities match, model probabilities match " begin

                s = vec([count_parameters(col) for col in m])
                indices = [findall(==(i), s) for i in 1:k]

                probs = vec([Distributions.pdf(d, col) for col in m])
                # pdf over all models sums to 1.0
                @test sum(probs) ≈ 1.0

                brute_force_incl_probs = [sum(probs[indices[i]]) for i in 1:k]
                efficient_incl_probs = [pdf_incl(d, i) for i in 1:k]

                # direct computation of inclusion probabilities (which is more efficient) equals brute force computation of inclusion probabilities
                @test efficient_incl_probs ≈ brute_force_incl_probs

                # no. equalities is insufficient for model probabilities for DPP
                if !(d isa AbstractProcessPartitionDistribution)

                    model_probs  = pdf_model.(Ref(d), 1:k)
                    model_counts = stirlings2.(k, 1:k)# .* EqualitySampler.count_combinations.(k, 1:k)

                    # dividing inclusion probabilities by model size frequency gives the model probabilities
                    @test efficient_incl_probs ./ model_counts ≈ model_probs
                else
                    @test_throws ArgumentError pdf_model(d, 1)
                    @test_throws ArgumentError logpdf_model(d, 1)
                end
            end

            @testset "simulated properties match theoretical results" begin

                # TODO: this test should NOT use reduce_model but just call rand!
                # samples = Distributions.rand(d, no_samples)

                samples = Distributions.rand(d, no_samples)
                if !(d isa DuplicatedPartitionDistribution)
                    @test all(col == EqualitySampler.reduce_model_2(col) for col in eachcol(samples))
                end

                # samples = Matrix{Int}(undef, k, no_samples)
                # ds = Distributions.sampler(d)
                # for i in axes(samples, 2)
                #     v = view(samples, :, i)
                #     Distributions.rand!(ds, v)
                    # v .= EqualitySampler.reduce_model(v)
                # end

                if !(d isa AbstractProcessPartitionDistribution)

                    empirical_model_probs_dict = EqualitySampler.compute_model_probs(samples, true)
                    sort!(empirical_model_probs_dict, by=x->count_parameters(x))

                    # empirical_model_probs_dict = EqualitySampler.empirical_model_probabilities(samples)

                    # add any missing models
                    # modelspace_distinct = PartitionSpace(k)
                    # if length(empirical_model_probs_dict) != length(modelspace_distinct)
                    #     for model in modelspace_distinct
                    #         get!(empirical_model_probs_dict, model, 0.0)
                    #         # # key = join(EqualitySampler.reduce_model(model))
                    #         # if !haskey(empirical_model_probs_dict, key)
                    #         #     empirical_model_probs_dict[key] = 0.0
                    #         # end
                    #     end
                    #     sort!(empirical_model_probs_dict, by=x->count_parameters(x))
                    # end
                    empirical_model_probs = collect(values(empirical_model_probs_dict))
                    expected_model_probs  = EqualitySampler.expected_model_probabilities(d)
                else # for the DirichletProcess this method doesn't exist (yet) so we brute force it
                    tmp                       = EqualitySampler.empirical_model_probabilities(samples)
                    empirical_model_probs     = collect(values(tmp))
                    expected_model_probs      = [exp(EqualitySampler.logpdf_model_distinct(d, reduce_model_dpp(parse.(Int, split(model, ""))))) for model in keys(tmp)]
                end

                empirical_inclusion_probs_dict = EqualitySampler.compute_incl_probs(samples; add_missing_inclusions = true)
                sort!(empirical_inclusion_probs_dict, byvalue = false)
                # empirical_inclusion_probs_dict = EqualitySampler.empirical_no_parameters_probabilities(samples)
                # if length(empirical_inclusion_probs_dict) != k + 1
                #     for i in 1:k
                #         if !haskey(empirical_inclusion_probs_dict, i)
                #             empirical_inclusion_probs_dict[i] = 0.0
                #         end
                #     end
                #     sort!(empirical_inclusion_probs_dict, by=values)
                # end

                empirical_inclusion_probs = collect(values(empirical_inclusion_probs_dict))
                expected_inclusion_probs  = EqualitySampler.expected_inclusion_probabilities(d)

                rtol = 0.15 + 0.02k # TODO: something better than this.
                @test isapprox(empirical_model_probs, expected_model_probs, rtol = rtol)
                @test isapprox(empirical_inclusion_probs, expected_inclusion_probs, rtol = rtol)

            end

            if !(d isa AbstractProcessPartitionDistribution)
                @testset "Batch computations match individual ones" begin

                    expected = EqualitySampler.log_expected_inclusion_probabilities(d)
                    observed = logpdf_incl.(Ref(d), 1:length(d))

                    @test isapprox(expected, observed)

                end
            end
        end
    end

    # @testset "logpdf_model_distinct works with DPP" begin

    #     for k in 2:10
    #         d = EqualitySampler.DirichletProcessPartitionDistribution(k, .5)
    #         m = Matrix(PartitionSpace(k))

    #         lpdfs = mapslices(x->logpdf_model_distinct(d, x), m, dims = 1)
    #         manual_sizes = [sort!(collect(values(StatsBase.countmap(x))); rev = true) for x in eachcol(m)]
    #         counts, sizes = EqualitySampler.count_set_partitions_given_partition_size(k)

    #         for i in eachindex(sizes)

    #             idx = findall(==(sizes[i]), manual_sizes)

    #             @test length(idx) == counts[i]
    #             @test sum(lpdfs[idx]) ≈ counts[i] * lpdfs[idx[1]]

    #         end

    #         parameter_counts = [count_parameters(x) for x in eachcol(m)]
    #         unique_parameter_counts = unique(parameter_counts)
    #         expected = [Statistics.mean(lpdfs[unique_parameter_counts_i .== parameter_counts]) for unique_parameter_counts_i in unique_parameter_counts]
    #         computed = logpdf_model_distinct.(Ref(d), unique_parameter_counts)
    #         @test expected ≈ computed

    #     end
    # end

    # TODO: merge this set with the previous
    @testset "logpdf_model_distinct is equal for value and models" begin

        for k in 3:6

            d_u  = UniformPartitionDistribution(k)
            d_bb = BetaBinomialPartitionDistribution(k, k, 1)
            models = Matrix(PartitionSpace(k))
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


