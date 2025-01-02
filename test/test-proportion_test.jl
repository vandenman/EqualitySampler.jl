using Test
import Random, Distributions, StatsBase as SB

nth(iter, idx) = first(Iterators.drop(iter, idx-1))
Random.seed!(42)
range_n_groups = 3:5
@testset "proportion_test_new" for n_groups in range_n_groups

    # n_groups = 5
    n_obs_per_group = 1000
    true_partition = rand(UniformPartitionDistribution(n_groups))

    # to test against
    ref_partition = EqualitySampler.reduce_model_2(true_partition)

    data_object = simulate_proportions(true_partition, fill(n_obs_per_group, n_groups))

    partition_prior = BetaBinomialPartitionDistribution(n_groups, 1, n_groups)

    enumerate_results  = proportion_test(data_object.n, data_object.k, EqualitySampler.Enumerate(), partition_prior, verbose = false)
    hpm_partition_enumerate = get_hpm_partition(enumerate_results)
    @test hpm_partition_enumerate == ref_partition

    # TODO: double check why these are different!
    # enumerate_results0 = proportions_enumerate(data_object.n, data_object.k, partition_prior)
    # enumerate_results.log_posterior_probs ≈ enumerate_results0.log_posterior_probs

    results_sample_integrated  = proportion_test(data_object.n, data_object.k, EqualitySampler.SampleIntegrated(), partition_prior, verbose = false)
    hpm_partition_sample_integrated = get_hpm_partition(results_sample_integrated)
    @test hpm_partition_sample_integrated == ref_partition

    results_enumerate_then_sample  = proportion_test(data_object.n, data_object.k, EqualitySampler.EnumerateThenSample(), partition_prior, verbose = false)
    hpm_partition_enumerate_then_sample = get_hpm_partition(results_enumerate_then_sample)
    @test hpm_partition_enumerate_then_sample == ref_partition

    results_sample = proportion_test(data_object.n, data_object.k, EqualitySampler.SampleRJMCMC(), partition_prior, verbose = false)
    hpm_partition_sample = get_hpm_partition(results_sample)
    @test hpm_partition_sample == ref_partition

    tol = 0.01n_groups

    @test SB.mean(results_sample.parameter_samples.θ_p_samples, dims = 2) ≈ data_object.p[data_object.partition] atol = tol


end
