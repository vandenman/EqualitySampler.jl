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

	StirlingStrategy,
	ExplicitStrategy,
	RecursiveStrategy,

	bellnumr,
	count_combinations,
	count_distinct_models,
	count_models_with_incl,
	count_equalities,
	reduce_model,
	expected_model_probabilities,
	expected_inclusion_counts,
	expected_inclusion_probabilities,
	log_expected_inclusion_counts,
	empirical_model_probabilities,
	empirical_inclusion_probabilities,
	generate_distinct_models,

	# based on Turing
	RandomProcessDistribution,
	expected_model_probabilities,
	expected_inclusion_probabilities
	#,
	# maybe these shouldn't be exported, at least, the lookup of the name of the turing samples shouldn't be done in there
	# compute_post_prob_eq,
	# compute_model_probs,
	# compute_incl_probs,
	# get_posterior_means_mu_sigma

include("combinatorialFunctions.jl")
include("conditionalUrnDistributions.jl")
include("normalLogLikelihood.jl")
include("helpers.jl")
include("PdfDirichletProcess.jl")

end

