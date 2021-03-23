module EqualitySampler

import Base: length
import Distributions, Random, LinearAlgebra, Memoize, SpecialFunctions
import StatsBase: countmap
import OrderedCollections: OrderedDict
import Turing, Turing.RandomMeasures

export
	NormalSuffStat,
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

	pdf_model,
	logpdf_model,
	pdf_incl,
	logpdf_incl,

	# based on Turing
	RandomProcessDistribution,
	expected_model_probabilities,
	expected_inclusion_probabilities,

	AbstractMvUrnDistribution,
	UniformMvUrnDistribution,
	BetaBinomialMvUrnDistribution,
	RandomProcessMvUrnDistribution

	#,
	# maybe these shouldn't be exported, at least, the lookup of the name of the turing samples shouldn't be done in there
	# compute_post_prob_eq,
	# compute_model_probs,
	# compute_incl_probs,
	# get_posterior_means_mu_sigma

include("combinatorialFunctions.jl")
include("multivariateUrnDistributions.jl")
include("conditionalUrnDistributions.jl")
include("normalLogLikelihood.jl")
include("helpers.jl")
include("PdfDirichletProcess.jl")


end

