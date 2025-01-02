using EqualitySampler, Test
import Random, Distributions, Statistics
import DataFrames as DF
import StatsBase as SB
import StatsModels
import LogarithmicNumbers
import OrderedCollections

@testset verbose = true "anova_test" begin

    drop_equalitysampler(x) = replace(string(x), "EqualitySampler." => "")
    drop_logarithmicnumbers(x) = replace(string(x), "LogarithmicNumbers." => "")
    Random.seed!(42)

    ks = 3:6
    @testset "model probabilities estimates for k = $k" for k in ks

        #k = 3
        n_obs_per_group = 1000
        true_partition = rand(UniformPartitionDistribution(k))

        # to test against
        ref_partition = EqualitySampler.reduce_model_2(true_partition)

        true_θ = .25 .* true_partition
        data_object = simulate_data_one_way_anova(k, n_obs_per_group, true_θ, true_partition, 0.25)
        data = data_object.data
        partition_prior = BetaBinomialPartitionDistribution(k, 1, k)

        eqprobs_dict    = Dict{String, Matrix{Float64}}()
        modelprobs_dict = Dict{String, AbstractDict}()
        @testset "method $(drop_equalitysampler(m)){$(drop_logarithmicnumbers(T))}" for
                m in (EqualitySampler.Enumerate, EqualitySampler.EnumerateThenSample, EqualitySampler.SampleIntegrated, EqualitySampler.SampleRJMCMC),
                T in (Float64, #=Brobdingnag.Brob{Float64}, =#LogarithmicNumbers.LogFloat64, LogarithmicNumbers.ULogFloat64)

            m_instance = m(integral_type = T)

            results = anova_test(data.y, data.g, m_instance, partition_prior, verbose = false)
            hpm_partition = get_hpm_partition(results)
            @test hpm_partition == ref_partition

            key = "$(drop_equalitysampler(m)){$(drop_logarithmicnumbers(T))}"
            eqprobs_dict[key]    = compute_post_prob_eq(results)
            modelprobs_dict[key] = compute_model_probs(results)

        end

        # m_instance = EqualitySampler.SampleRJMCMC(integral_type = Float64, iter = 100_000, split_merge_prob = 1.0)
        # results = anova_test(data.y, data.g, m_instance, partition_prior, verbose = false)
        # hpm_partition = get_hpm_partition(results)
        # @test hpm_partition == ref_partition


        ref_method = "Enumerate{Float64}"
        for (dict, nm, f) in (
            (eqprobs_dict, "eq_probs", identity),
            (modelprobs_dict, "model_probs", collect ∘ values)
        )
            ref_vals   = dict[ref_method]
            @testset "$nm $(drop_equalitysampler(m)){$T}" for
                m in (EqualitySampler.Enumerate, EqualitySampler.EnumerateThenSample, EqualitySampler.SampleIntegrated, EqualitySampler.SampleRJMCMC),
                T in (Float64, #=Brobdingnag.Brob{Float64}, =#LogarithmicNumbers.LogFloat64, LogarithmicNumbers.ULogFloat64)

                key = "$(drop_equalitysampler(m)){$(drop_logarithmicnumbers(T))}"
                @test f(ref_vals) ≈ f(dict[ref_method])# atol = 1e-1
            end
        end


    end


    @testset "parameter estimates for k = $k" for k in 2:4:12

        tol = 0.01k

        partition = EqualitySampler.reduce_model_2(rand(UniformPartitionDistribution(k)))
        θ = 1.5 .* collect(partition)
        θ .-= SB.mean(θ)
        sample_sizes = rand(10:10:100, k)
        μ = rand(Distributions.Uniform(10, 20))
        sim_obj = simulate_data_one_way_anova(
            k,
            sample_sizes .* 20_000, θ, partition, 12.34, 1.65);

        obj_ss = EqualitySampler.extract_suffstats_one_way_anova(sim_obj.data.y, sim_obj.data.g)

        sampling_methods = (
            EqualitySampler.SampleRJMCMC(iter = 20_000, fullmodel_only = true),
            EqualitySampler.EnumerateThenSample(iter = 20_000),
            EqualitySampler.SampleRJMCMC(iter = 20_000)
        )

        true_values = [sim_obj.true_values.μ; sim_obj.true_values.σ; sim_obj.true_values.θ]
        parameter   = reduce(vcat, fill.([:μ, :σ, :θₛ ], (1, 1, k)))
        true_eqs = [partition[i] == partition[j] && i > j for i in eachindex(partition), j in eachindex(partition)]

        @testset "method $m" for m in sampling_methods

            samps = anova_test(obj_ss, m, UniformPartitionDistribution(length(partition)), verbose = false, threaded = true)
            means_df = DF.DataFrame(
                post_mean  = [SB.mean(samps.parameter_samples.μ); SB.mean(sqrt, samps.parameter_samples.σ²); vec(SB.mean(samps.parameter_samples.θ_cp, dims = 2))],
                true_value = true_values,
                parameter  = parameter
            )

            means_df_grp = DF.groupby(means_df, :parameter)

            @test means_df_grp[(:μ, ) ].post_mean ≈ means_df_grp[(:μ, ) ].true_value atol = tol
            @test means_df_grp[(:σ, ) ].post_mean ≈ means_df_grp[(:σ, ) ].true_value atol = tol
            @test means_df_grp[(:θₛ, )].post_mean ≈ means_df_grp[(:θₛ, )].true_value atol = tol

            cell_means_df_eq = DF.DataFrame(
                obs_mean  = obj_ss.y_mean_by_group,
                est_value = vec(SB.mean(samps.parameter_samples.μ' .+ samps.parameter_samples.θ_cp, dims = 2)),
            )

            @test cell_means_df_eq.obs_mean ≈ cell_means_df_eq.est_value atol = tol

            if !isempty(samps.partition_samples)

                model_avg_eq_probs = compute_post_prob_eq(samps)
                model_avg_eq_probs_error = abs.(true_eqs .- model_avg_eq_probs)
                @test maximum(model_avg_eq_probs_error) < tol

            end

        end
    end

    # samps_full = anova_test(obj_ss,
        #     EqualitySampler.SampleRJMCMC(iter = 50_000, fullmodel_only = true),
        #     UniformPartitionDistribution(length(partition)),
        #     verbose = false
        # )

        # means_df = DF.DataFrame(
        #     post_mean  = [SB.mean(samps_full.parameter_samples.μ); SB.mean(sqrt, samps_full.σ²_samples); vec(SB.mean(samps_full.parameter_samples.θ_cp, dims = 2))],
        #     true_value = [sim_obj.true_values.μ; sim_obj.true_values.σ; sim_obj.true_values.θ],
        #     parameter  = reduce(vcat, fill.([:μ, :σ, :θₛ ], (1, 1, k)))
        # )

        # means_df_grp = DF.groupby(means_df, :parameter)

        # @test means_df_grp[(:μ, ) ].post_mean ≈ means_df_grp[(:μ, ) ].true_value atol = tol
        # @test means_df_grp[(:σ, ) ].post_mean ≈ means_df_grp[(:σ, ) ].true_value atol = tol
        # @test means_df_grp[(:θₛ, )].post_mean ≈ means_df_grp[(:θₛ, )].true_value atol = tol

        # # one2one =  AoG.mapping([0], [1]) * AoG.visual(CM.ABLines, color=:grey, linestyle=:dash)
        # # d = AoG.data(means_df)
        # # one2one + d * AoG.mapping(:true_value, :post_mean, color = :parameter, marker = :parameter) * (AoG.visual(;markersize = 15)) |> AoG.draw

        # cell_means_df_eq = DF.DataFrame(
        #     obs_mean  = obj_ss.y_mean_by_group,
        #     est_value = vec(SB.mean(samps_full.parameter_samples.μ' .+ samps_full.parameter_samples.θ_cp, dims = 2)),
        # )

        # @test cell_means_df_eq.obs_mean ≈ cell_means_df_eq.est_value atol = tol

        # # d = AoG.data(cell_means_df_eq)
        # # one2one + d * AoG.mapping(:obs_mean, :est_value) * (AoG.visual(;markersize = 15)) |> AoG.draw

        # if k <= 4

        #     samps_eq = anova_test(obj_ss,
        #         EqualitySampler.EnumerateThenSample(iter = 100_000),
        #         UniformPartitionDistribution(k),
        #         verbose = false
        #     )

        #     true_eqs = [partition[i] == partition[j] && i > j for i in eachindex(partition), j in eachindex(partition)]
        #     model_avg_eq_probs = compute_post_prob_eq(samps_eq.partition_samples')

        #     model_avg_eq_probs_error = abs.(true_eqs .- model_avg_eq_probs)

        #     @test maximum(model_avg_eq_probs_error) < tol

        #     means_df_eq = DF.DataFrame(
        #         post_mean  = [SB.mean(samps_eq.parameter_samples.μ); SB.mean(sqrt, samps_eq.σ²_samples); vec(SB.mean(samps_eq.parameter_samples.θ_cp, dims = 2))],
        #         true_value = [sim_obj.true_values.μ; sim_obj.true_values.σ; sim_obj.true_values.θ],
        #         parameter  = reduce(vcat, fill.([:μ, :σ, :θₛ ], (1, 1, k)))
        #     )

        #     means_df_eq_grp = DF.groupby(means_df_eq, :parameter)

        #     @test means_df_eq_grp[(:μ, ) ].post_mean ≈ means_df_eq_grp[(:μ, ) ].true_value atol = tol
        #     @test means_df_eq_grp[(:σ, ) ].post_mean ≈ means_df_eq_grp[(:σ, ) ].true_value atol = tol
        #     @test means_df_eq_grp[(:θₛ, )].post_mean ≈ means_df_eq_grp[(:θₛ, )].true_value atol = tol

        #     # one2one =  AoG.mapping([0], [1]) * AoG.visual(CM.ABLines, color=:grey, linestyle=:dash)
        #     # d = AoG.data(means_df_eq)
        #     # one2one + d * AoG.mapping(:true_value, :post_mean, color = :parameter, marker = :parameter) * (AoG.visual(;markersize = 15)) |> AoG.draw

        #     cell_means_df_eq = DF.DataFrame(
        #         obs_mean  = obj_ss.y_mean_by_group,
        #         est_value = vec(SB.mean(samps_eq.parameter_samples.μ' .+ samps_eq.parameter_samples.θ_cp, dims = 2)),
        #     )

        #     @test cell_means_df_eq.obs_mean ≈ cell_means_df_eq.est_value atol = tol

        # end

        # samps_eq = anova_test(obj_ss,
        #     EqualitySampler.SampleRJMCMC(iter = 100_000),
        #     UniformPartitionDistribution(k),
        #     verbose = false
        # )

        # true_eqs = [partition[i] == partition[j] && i > j for i in eachindex(partition), j in eachindex(partition)]
        # model_avg_eq_probs = compute_post_prob_eq(samps_eq.partition_samples')

        # model_avg_eq_probs_error = abs.(true_eqs .- model_avg_eq_probs)

        # @test maximum(model_avg_eq_probs_error) < tol

        # means_df_eq = DF.DataFrame(
        #     post_mean  = [SB.mean(samps_eq.parameter_samples.μ); SB.mean(sqrt, samps_eq.σ²_samples); vec(SB.mean(samps_eq.parameter_samples.θ_cp, dims = 2))],
        #     true_value = [sim_obj.true_values.μ; sim_obj.true_values.σ; sim_obj.true_values.θ],
        #     parameter  = reduce(vcat, fill.([:μ, :σ, :θₛ ], (1, 1, k)))
        # )

        # means_df_eq_grp = DF.groupby(means_df_eq, :parameter)

        # @test means_df_eq_grp[(:μ, ) ].post_mean ≈ means_df_eq_grp[(:μ, ) ].true_value atol = tol
        # @test means_df_eq_grp[(:σ, ) ].post_mean ≈ means_df_eq_grp[(:σ, ) ].true_value atol = tol
        # @test means_df_eq_grp[(:θₛ, )].post_mean ≈ means_df_eq_grp[(:θₛ, )].true_value atol = tol

        # # one2one =  AoG.mapping([0], [1]) * AoG.visual(CM.ABLines, color=:grey, linestyle=:dash)
        # # d = AoG.data(means_df_eq)
        # # one2one + d * AoG.mapping(:true_value, :post_mean, color = :parameter, marker = :parameter) * (AoG.visual(;markersize = 15)) |> AoG.draw

        # cell_means_df_eq = DF.DataFrame(
        #     obs_mean  = obj_ss.y_mean_by_group,
        #     est_value = vec(SB.mean(samps_eq.parameter_samples.μ' .+ samps_eq.parameter_samples.θ_cp, dims = 2)),
        # )

        # @test cell_means_df_eq.obs_mean ≈ cell_means_df_eq.est_value atol = tol

        # # d = AoG.data(cell_means_df_eq)
        # # one2one + d * AoG.mapping(:obs_mean, :est_value) * (AoG.visual(;markersize = 15)) |> AoG.draw
    # end

    @testset "initial partition" begin

        k = 6
        tol = 0.01k

        partition = EqualitySampler.reduce_model_2(rand(UniformPartitionDistribution(k)))
        θ = 1.5 .* collect(partition)
        θ .-= SB.mean(θ)
        sample_sizes = rand(10:10:100, k)
        μ = rand(Distributions.Uniform(10, 20))
        sim_obj = simulate_data_one_way_anova(
            k,
            sample_sizes .* 20_000, θ, partition, 12.34, 1.65);

        obj_ss = EqualitySampler.extract_suffstats_one_way_anova(sim_obj.data.y, sim_obj.data.g)
        samps_full = anova_test(obj_ss,
            EqualitySampler.SampleRJMCMC(iter = 10_000, fullmodel_only = true, initial_partition = partition),
            UniformPartitionDistribution(length(partition)),
            verbose = false
        )

        means_df = DF.DataFrame(
            post_mean  = [SB.mean(samps_full.parameter_samples.μ); SB.mean(sqrt, samps_full.parameter_samples.σ²); vec(SB.mean(samps_full.parameter_samples.θ_cp, dims = 2))],
            true_value = [sim_obj.true_values.μ; sim_obj.true_values.σ; sim_obj.true_values.θ],
            parameter  = reduce(vcat, fill.([:μ, :σ, :θₛ ], (1, 1, k)))
        )

        means_df_grp = DF.groupby(means_df, :parameter)

        @test means_df_grp[(:μ, ) ].post_mean ≈ means_df_grp[(:μ, ) ].true_value atol = tol
        @test means_df_grp[(:σ, ) ].post_mean ≈ means_df_grp[(:σ, ) ].true_value atol = tol
        @test means_df_grp[(:θₛ, )].post_mean ≈ means_df_grp[(:θₛ, )].true_value atol = tol

        # one2one =  AoG.mapping([0], [1]) * AoG.visual(CM.ABLines, color=:grey, linestyle=:dash)
        # d = AoG.data(means_df)
        # one2one + d * AoG.mapping(:true_value, :post_mean, color = :parameter, marker = :parameter) * (AoG.visual(;markersize = 15)) |> AoG.draw

        cell_means_df_eq = DF.DataFrame(
            obs_mean  = obj_ss.y_mean_by_group,
            est_value = vec(SB.mean(samps_full.parameter_samples.μ' .+ samps_full.parameter_samples.θ_cp, dims = 2)),
        )

        @test cell_means_df_eq.obs_mean ≈ cell_means_df_eq.est_value atol = tol

        # d = AoG.data(cell_means_df_eq)
        # one2one + d * AoG.mapping(:obs_mean, :est_value) * (AoG.visual(;markersize = 15)) |> AoG.draw

        samps_eq = anova_test(obj_ss,
            EqualitySampler.SampleRJMCMC(iter = k*10_000, initial_partition = partition),
            UniformPartitionDistribution(k),
            verbose = false,
            threaded = false
        )

        true_eqs = [partition[i] == partition[j] && i > j for i in eachindex(partition), j in eachindex(partition)]
        model_avg_eq_probs = compute_post_prob_eq(samps_eq)

        model_avg_eq_probs_error = abs.(true_eqs .- model_avg_eq_probs)

        @test maximum(model_avg_eq_probs_error) < tol

        means_df_eq = DF.DataFrame(
            post_mean  = [SB.mean(samps_eq.parameter_samples.μ); SB.mean(sqrt, samps_eq.parameter_samples.σ²); vec(SB.mean(samps_eq.parameter_samples.θ_cp, dims = 2))],
            true_value = [sim_obj.true_values.μ; sim_obj.true_values.σ; sim_obj.true_values.θ],
            parameter  = reduce(vcat, fill.([:μ, :σ, :θₛ ], (1, 1, k)))
        )

        means_df_eq_grp = DF.groupby(means_df_eq, :parameter)

        @test means_df_eq_grp[(:μ, ) ].post_mean ≈ means_df_eq_grp[(:μ, ) ].true_value atol = tol
        @test means_df_eq_grp[(:σ, ) ].post_mean ≈ means_df_eq_grp[(:σ, ) ].true_value atol = tol
        @test means_df_eq_grp[(:θₛ, )].post_mean ≈ means_df_eq_grp[(:θₛ, )].true_value atol = tol

        # one2one =  AoG.mapping([0], [1]) * AoG.visual(CM.ABLines, color=:grey, linestyle=:dash)
        # d = AoG.data(means_df_eq)
        # one2one + d * AoG.mapping(:true_value, :post_mean, color = :parameter, marker = :parameter) * (AoG.visual(;markersize = 15)) |> AoG.draw

        cell_means_df_eq = DF.DataFrame(
            obs_mean  = obj_ss.y_mean_by_group,
            est_value = vec(SB.mean(samps_eq.parameter_samples.μ' .+ samps_eq.parameter_samples.θ_cp, dims = 2)),
        )

        @test cell_means_df_eq.obs_mean ≈ cell_means_df_eq.est_value atol = tol

        # d = AoG.data(cell_means_df_eq)
        # one2one + d * AoG.mapping(:obs_mean, :est_value) * (AoG.visual(;markersize = 15)) |> AoG.draw
    end

    @testset "StatsModels formula input" begin

        k = 4
        tol = 0.01k

        partition = EqualitySampler.reduce_model_2(rand(UniformPartitionDistribution(k)))
        θ = 1.5 .* collect(partition)
        θ .-= SB.mean(θ)
        sample_sizes = rand(10:10:100, k)
        μ = rand(Distributions.Uniform(10, 20))
        sim_obj = simulate_data_one_way_anova(
            k,
            sample_sizes .* 1_000, θ, partition, 12.34, 1.65);
        obj_ss = EqualitySampler.extract_suffstats_one_way_anova(sim_obj.data.y, sim_obj.data.g)

        df = DF.DataFrame(
            outcome = sim_obj.data.y,
            grouping = reduce(vcat, [fill(i, length(g)) for (i, g) in enumerate(sim_obj.data.g)])
        )

        samps_full = anova_test(
            StatsModels.@formula(outcome ~ 1 + grouping), df,
            EqualitySampler.SampleRJMCMC(iter = 10_000, fullmodel_only = true),
            UniformPartitionDistribution(length(partition)),
            verbose = false
        )

        means_df = DF.DataFrame(
            post_mean  = [SB.mean(samps_full.parameter_samples.μ); SB.mean(sqrt, samps_full.parameter_samples.σ²); vec(SB.mean(samps_full.parameter_samples.θ_cp, dims = 2))],
            true_value = [sim_obj.true_values.μ; sim_obj.true_values.σ; sim_obj.true_values.θ],
            parameter  = reduce(vcat, fill.([:μ, :σ, :θₛ ], (1, 1, k)))
        )

        means_df_grp = DF.groupby(means_df, :parameter)

        @test means_df_grp[(:μ, ) ].post_mean ≈ means_df_grp[(:μ, ) ].true_value atol = tol
        @test means_df_grp[(:σ, ) ].post_mean ≈ means_df_grp[(:σ, ) ].true_value atol = tol
        @test means_df_grp[(:θₛ, )].post_mean ≈ means_df_grp[(:θₛ, )].true_value atol = tol

        cell_means_df_eq = DF.DataFrame(
            obs_mean  = obj_ss.y_mean_by_group,
            est_value = vec(SB.mean(samps_full.parameter_samples.μ' .+ samps_full.parameter_samples.θ_cp, dims = 2)),
        )

        @test cell_means_df_eq.obs_mean ≈ cell_means_df_eq.est_value atol = tol


        samps_eq = anova_test(
            StatsModels.@formula(outcome ~ 1 + grouping), df,
            EqualitySampler.SampleRJMCMC(iter = k*20_000, initial_partition = partition),
            UniformPartitionDistribution(k),
            verbose = false
        )

        true_eqs = [partition[i] == partition[j] && i > j for i in eachindex(partition), j in eachindex(partition)]
        model_avg_eq_probs = compute_post_prob_eq(samps_eq)

        model_avg_eq_probs_error = abs.(true_eqs .- model_avg_eq_probs)

        @test maximum(model_avg_eq_probs_error) < tol

        means_df_eq = DF.DataFrame(
            post_mean  = [SB.mean(samps_eq.parameter_samples.μ); SB.mean(sqrt, samps_eq.parameter_samples.σ²); vec(SB.mean(samps_eq.parameter_samples.θ_cp, dims = 2))],
            true_value = [sim_obj.true_values.μ; sim_obj.true_values.σ; sim_obj.true_values.θ],
            parameter  = reduce(vcat, fill.([:μ, :σ, :θₛ ], (1, 1, k)))
        )

        means_df_eq_grp = DF.groupby(means_df_eq, :parameter)

        @test means_df_eq_grp[(:μ, ) ].post_mean ≈ means_df_eq_grp[(:μ, ) ].true_value atol = tol
        @test means_df_eq_grp[(:σ, ) ].post_mean ≈ means_df_eq_grp[(:σ, ) ].true_value atol = tol
        @test means_df_eq_grp[(:θₛ, )].post_mean ≈ means_df_eq_grp[(:θₛ, )].true_value atol = tol

        # one2one =  AoG.mapping([0], [1]) * AoG.visual(CM.ABLines, color=:grey, linestyle=:dash)
        # d = AoG.data(means_df_eq)
        # one2one + d * AoG.mapping(:true_value, :post_mean, color = :parameter, marker = :parameter) * (AoG.visual(;markersize = 15)) |> AoG.draw

        cell_means_df_eq = DF.DataFrame(
            obs_mean  = obj_ss.y_mean_by_group,
            est_value = vec(SB.mean(samps_eq.parameter_samples.μ' .+ samps_eq.parameter_samples.θ_cp, dims = 2)),
        )

        @test cell_means_df_eq.obs_mean ≈ cell_means_df_eq.est_value atol = tol


    end

    @testset "threaded sampling works" begin

        k = 3
        tol = 0.01k

        partition = EqualitySampler.reduce_model_2(rand(UniformPartitionDistribution(k)))
        θ = 1.5 .* collect(partition)
        θ .-= SB.mean(θ)
        sample_sizes = rand(100:10:200, k)
        μ = rand(Distributions.Uniform(10, 20))
        sim_obj = simulate_data_one_way_anova(
            k,
            sample_sizes .* 1_000, θ, partition, 12.34, 1.65);
        obj_ss = EqualitySampler.extract_suffstats_one_way_anova(sim_obj.data.y, sim_obj.data.g)

        df = DF.DataFrame(
            outcome = sim_obj.data.y,
            grouping = reduce(vcat, [fill(i, length(g)) for (i, g) in enumerate(sim_obj.data.g)])
        )

        samps_full = anova_test(
            StatsModels.@formula(outcome ~ 1 + grouping), df,
            EqualitySampler.SampleRJMCMC(iter = 10_000, fullmodel_only = true),
            UniformPartitionDistribution(length(partition)),
            verbose = false
        )

        means_df = DF.DataFrame(
            post_mean  = [SB.mean(samps_full.parameter_samples.μ); SB.mean(sqrt, samps_full.parameter_samples.σ²); vec(SB.mean(samps_full.parameter_samples.θ_cp, dims = 2))],
            true_value = [sim_obj.true_values.μ; sim_obj.true_values.σ; sim_obj.true_values.θ],
            parameter  = reduce(vcat, fill.([:μ, :σ, :θₛ ], (1, 1, k)))
        )

        means_df_grp = DF.groupby(means_df, :parameter)

        @test means_df_grp[(:μ, ) ].post_mean ≈ means_df_grp[(:μ, ) ].true_value atol = tol
        @test means_df_grp[(:σ, ) ].post_mean ≈ means_df_grp[(:σ, ) ].true_value atol = tol
        @test means_df_grp[(:θₛ, )].post_mean ≈ means_df_grp[(:θₛ, )].true_value atol = tol

        cell_means_df_eq = DF.DataFrame(
            obs_mean  = obj_ss.y_mean_by_group,
            est_value = vec(SB.mean(samps_full.parameter_samples.μ' .+ samps_full.parameter_samples.θ_cp, dims = 2)),
        )

        @test cell_means_df_eq.obs_mean ≈ cell_means_df_eq.est_value atol = tol


        samps_eq = anova_test(
            StatsModels.@formula(outcome ~ 1 + grouping), df,
            EqualitySampler.SampleRJMCMC(iter = k * 30_000, initial_partition = partition),
            UniformPartitionDistribution(k),
            verbose = false,
            threaded = true
        )

        true_eqs = [partition[i] == partition[j] && i > j for i in eachindex(partition), j in eachindex(partition)]
        model_avg_eq_probs = compute_post_prob_eq(samps_eq)


        model_avg_eq_probs_error = abs.(true_eqs .- model_avg_eq_probs)

        @test maximum(model_avg_eq_probs_error) < tol

        means_df_eq = DF.DataFrame(
            post_mean  = [SB.mean(samps_eq.parameter_samples.μ); SB.mean(sqrt, samps_eq.parameter_samples.σ²); vec(SB.mean(samps_eq.parameter_samples.θ_cp, dims = 2))],
            true_value = [sim_obj.true_values.μ; sim_obj.true_values.σ; sim_obj.true_values.θ],
            parameter  = reduce(vcat, fill.([:μ, :σ, :θₛ ], (1, 1, k)))
        )

        means_df_eq_grp = DF.groupby(means_df_eq, :parameter)

        @test means_df_eq_grp[(:μ, ) ].post_mean ≈ means_df_eq_grp[(:μ, ) ].true_value atol = tol
        @test means_df_eq_grp[(:σ, ) ].post_mean ≈ means_df_eq_grp[(:σ, ) ].true_value atol = tol
        @test means_df_eq_grp[(:θₛ, )].post_mean ≈ means_df_eq_grp[(:θₛ, )].true_value atol = tol

        # one2one =  AoG.mapping([0], [1]) * AoG.visual(CM.ABLines, color=:grey, linestyle=:dash)
        # d = AoG.data(means_df_eq)
        # one2one + d * AoG.mapping(:true_value, :post_mean, color = :parameter, marker = :parameter) * (AoG.visual(;markersize = 15)) |> AoG.draw

        cell_means_df_eq = DF.DataFrame(
            obs_mean  = obj_ss.y_mean_by_group,
            est_value = vec(SB.mean(samps_eq.parameter_samples.μ' .+ samps_eq.parameter_samples.θ_cp, dims = 2)),
        )

        @test cell_means_df_eq.obs_mean ≈ cell_means_df_eq.est_value atol = tol

    end

end
