module EqualitySampler

import Base: length
import Distributions, Random, LinearAlgebra, SpecialFunctions, Optim
import StatsBase: countmap
import OrderedCollections: OrderedDict
import Turing, Turing.RandomMeasures
import Bijectors
import Combinatorics
import Statistics

# for BayesFactor
import QuadGK

export
	NormalSuffStat,
	MvNormalSuffStat,
	logpdf_mv_normal_suffstat,
	MvNormalDenseSuffStat,
	MvNormalCholDenseSuffStat,
	get_normal_dense_suff_stats,
	get_normal_dense_chol_suff_stats,
	logpdf_mv_normal_chol_suffstat,
	AbstractConditionalUrnDistribution,
	UniformConditionalUrnDistribution,
	BetaBinomialConditionalUrnDistribution,
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

	count_combinations,
	count_distinct_models,
	count_models_with_incl,
	count_distinct_models_with_incl,
	log_count_distinct_models_with_incl,
	count_equalities,
	reduce_model,
	reduce_model_dpp,
	# reduce_model!,
	expected_model_probabilities,
	expected_inclusion_counts,
	expected_inclusion_probabilities,
	log_expected_inclusion_counts,
	log_expected_inclusion_probabilities,
	empirical_model_probabilities,
	empirical_inclusion_probabilities,
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

	bayes_factor_one_way_anova,

	# Jeffreys priors for variances and sds
	JeffreysPriorStandardDeviation,
	JeffreysPriorVariance

	#,
	# maybe these shouldn't be exported, at least, the lookup of the name of the turing samples shouldn't be done in there
	# compute_post_prob_eq,
	# compute_model_probs,
	# compute_incl_probs,
	# get_posterior_means_mu_sigma

include("stirling_helpers.jl")
include("stirling1.jl")
include("stirling2.jl")
include("rstirling2.jl")
include("bellnumr.jl")
include("lookupTablesStirlingRBellnumbers.jl")
include("generateModelSpace.jl")
include("combinatorialFunctions.jl")
include("multivariateUrnDistributions.jl")
include("conditionalUrnDistributions.jl")
include("normalLogLikelihood.jl")
include("helpers.jl")
include("PdfDirichletProcess.jl")
include("bayesFactors.jl")
include("PartitionSampler.jl")
include("findDPPalpha.jl")
include("JeffreysPrior.jl")


end

