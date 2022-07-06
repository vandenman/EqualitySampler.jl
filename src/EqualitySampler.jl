module EqualitySampler

# stdlib
import Base: length
import LinearAlgebra, Random, Statistics

import
	Bijectors,
	Combinatorics,
	Distributions,
	OrderedCollections,
	StatsBase,
	SpecialFunctions,
	Turing,
	Turing.RandomMeasures

using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES

export
	NormalSuffStat,
	MvNormalSuffStat,
	logpdf_mv_normal_suffstat,
	MvNormalDenseSuffStat,
	MvNormalCholDenseSuffStat,
	get_normal_dense_suff_stats,
	get_normal_dense_chol_suff_stats,
	logpdf_mv_normal_chol_suffstat,
	logpdf_mv_normal_precision_chol_suffstat,
	AbstractConditionalUrnDistribution,
	UniformConditionalUrnDistribution,
	BetaBinomialConditionalUrnDistribution,
	CustomInclusionMvUrnDistribution,
	stirlings2,
	stirlings2r,
	logstirlings2,
	logstirlings2r,

	unsignedstirlings1,
	logunsignedstirlings1,

	StirlingStrategy,
	ExplicitStrategy,
	RecursiveStrategy,

	bellnumr,
	logbellnumr,
	bellnum,

	count_combinations,
	count_distinct_models,
	count_models_with_no_equalities,
	count_distinct_models_with_no_equalities,
	log_count_distinct_models_with_no_equalities,
	count_models_with_no_parameters,
	count_distinct_models_with_no_parameters,
	log_count_distinct_models_with_no_parameters,
	count_equalities,
	count_parameters,
	reduce_model,
	reduce_model_dpp,
	# reduce_model!,
	expected_model_probabilities,
	expected_inclusion_counts,
	expected_inclusion_probabilities,
	log_expected_equality_counts,
	log_expected_inclusion_probabilities,
	empirical_model_probabilities,
	empirical_equality_probabilities,
	empirical_no_parameters_probabilities,
	generate_distinct_models,
	generate_all_models,
	count_set_partitions_given_partition_size,

	pdf_model,
	logpdf_model,
	pdf_model_distinct,
	logpdf_model_distinct,
	pdf_incl,
	logpdf_incl,

	# based on Turing
	RandomProcessDistribution,
	expected_model_probabilities,
	expected_inclusion_probabilities,

	AbstractMvUrnDistribution,
	UniformMvUrnDistribution,
	BetaBinomialMvUrnDistribution,
	BetaBinomialProcessMvUrnDistribution,
	RandomProcessMvUrnDistribution,
	DirichletProcessMvUrnDistribution,
	PartitionSampler,

	dpp_find_α,

	# bayes_factor_one_way_anova,

	# Jeffreys priors for variances and sds
	JeffreysPriorStandardDeviation,
	JeffreysPriorVariance

	#,
	# maybe these shouldn't be exported, at least, the lookup of the name of the turing samples shouldn't be done in there
	# compute_post_prob_eq,
	# compute_model_probs,
	# compute_incl_probs,
	# get_posterior_means_mu_sigma

include("special_functions.jl")
include("stirling_helpers.jl")
include("stirling1.jl")
include("stirling2.jl")
include("rstirling2.jl")
include("bellnumr.jl")
include("lookup_tables_stirling_r_bellnumbers.jl")
include("partition_shortcuts.jl")
include("generate_model_space.jl")
include("combinatorics.jl")
include("multivariate_urn_distributions.jl")
include("conditional_urn_distributions.jl")
include("normal_loglikelihood.jl")
include("helpers.jl")
include("partition_sampler.jl")
include("dpp_find_alpha.jl")
include("jeffreys_prior.jl")

include("simulations/Simulations.jl")
import .Simulations: anova_test, proportion_test
export
	anova_test,
	proportion_test




end

