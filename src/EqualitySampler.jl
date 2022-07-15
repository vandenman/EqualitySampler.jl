module EqualitySampler

# stdlib
import Base: length
import LinearAlgebra, Random, Statistics

import
	Bijectors,
	Combinatorics,
	Distributions,
	OrderedCollections,
	PDMats,
	StatsBase,
	SpecialFunctions,
	Turing,
	Turing.RandomMeasures

using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES

export
	#=
		TODO:
		write and export one function that computes the likelihood using Distributions.suffstats(MvNormal, x) and Distributions.suffstats(Normal, x)
		perhaps skip/ remove logpdf_mv_normal_chol_suffstat and friends (constructing the full matrix is not super expensive)
		also use PDMats.invquad and friends
	=#
	get_normal_dense_chol_suff_stats,
	logpdf_mv_normal_chol_suffstat,
	loglikelihood_suffstats,

	stirlings2,
	stirlings2r,
	logstirlings2,
	logstirlings2r,

	unsignedstirlings1,
	logunsignedstirlings1,

	# TODO: do not export these
	StirlingStrategy,
	ExplicitStrategy,
	RecursiveStrategy,

	bellnum,
	bellnumr,
	logbellnumr,

	# TODO: refactor all the count_xxx methods to a few sensible ones
	count_combinations,
	count_distinct_models,
	count_models_with_no_equalities,
	count_models_with_no_parameters,
	count_distinct_models_with_no_parameters,
	count_distinct_models_with_no_equalities,
	count_equalities,
	count_parameters,

	log_count_distinct_models_with_no_equalities,
	log_count_distinct_models_with_no_parameters,

	# TODO: unify these two. if reduce_model_dpp is only required for testing only define it there
	reduce_model,
	reduce_model_dpp,
	# reduce_model!,

	# TODO: refactor and remove whatever is possible
	expected_model_probabilities,
	expected_inclusion_counts,
	expected_inclusion_probabilities,

	log_expected_equality_counts,
	log_expected_inclusion_probabilities,

	empirical_model_probabilities,
	empirical_equality_probabilities,
	empirical_no_parameters_probabilities,


	# TODO: deprecate in favor of the iterator
	generate_distinct_models,
	generate_all_models,

	# TODO: group with other count_xxx methods
	count_set_partitions_given_partition_size,

	# TODO: group with distribution functions
	pdf_model,
	pdf_model_distinct,
	pdf_incl,

	logpdf_model,
	logpdf_model_distinct,
	logpdf_incl,

	# TODO
	# based on Turing

	RandomProcessDistribution, # TODO: different grouping

	# TODO: different grouping
	expected_model_probabilities,
	expected_inclusion_probabilities,

	AbstractMvUrnDistribution,
	UniformMvUrnDistribution,
	BetaBinomialMvUrnDistribution,
	CustomInclusionMvUrnDistribution,
	RandomProcessMvUrnDistribution,
	DirichletProcessMvUrnDistribution,
	BetaBinomialProcessMvUrnDistribution, # TODO: <- wasn't this a failed experiment? if so remove!

	# TODO: better name
	PartitionSampler,

	# TODO: different grouping
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

