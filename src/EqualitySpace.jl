module EqualitySpace

import Distributions, Random, LinearAlgebra, Memoize
import StatsBase: countmap

export
	NormalSuffStat,
	AbstractConditionalUrnDistribution,
	UniformConditionalUrnDistribution,
	BetaBinomialConditionalUrnDistribution,
	stirlings2,
	stirlings2r,
	bellnumr,
	count_combinations,
	count_distinct_models,
	count_models_with_incl,
	reduce_model,
	expected_model_probabilities,
	expected_inclusion_counts,
	expected_inclusion_probabilities,
	empirical_model_probabilities,
	empirical_inclusion_probabilities

include("combinatorialFunctions.jl")
include("conditionalUrnDistributions.jl")
include("normalLogLikelihood.jl")
include("helperfunctions.jl")

end

