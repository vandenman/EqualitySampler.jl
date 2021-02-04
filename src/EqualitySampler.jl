module EqualitySampler

import Distributions, Random, LinearAlgebra, Memoize, SpecialFunctions
import StatsBase: countmap
import OrderedCollections: OrderedDict

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
	empirical_model_probabilities,
	empirical_inclusion_probabilities,
	generate_distinct_models

include("combinatorialFunctions.jl")
include("conditionalUrnDistributions.jl")
include("normalLogLikelihood.jl")
include("helperfunctions.jl")

end

