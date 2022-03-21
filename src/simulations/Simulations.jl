module Simulations

using ..EqualitySampler

# TODO: should we just do using Turing?

# stdlib
import
	LinearAlgebra,
	Logging,
	Random

import
	Statistics: mean, var

# other dependencies
import
	AbstractMCMC,
	CategoricalArrays,
	DataFrames,
	Distributions,
	DistributionsAD,
	DynamicPPL,
	FillArrays,
	GLM,
	HypergeometricFunctions,
	IOCapture,
	MCMCChains,
	OrderedCollections,
	QuadGK,
	SpecialFunctions,
	StatsBase,
	StatsModels,
	Suppressor,
	Turing

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF, TYPEDFIELDS

export
	MCMCSettings,
	SimpleDataSet,
	simulate_data_one_way_anova,
	normalize_θ,
	compute_post_prob_eq,
	anova_test,
	westfall_test,
	proportion_test#,
	# variance_test

include("MCMCSettings.jl")
include("brute_force_epsilon.jl")
include("turing_helpers.jl")
include("proportion_functions.jl")
include("simpledataset.jl")
include("anova_functions.jl")
include("logpdf_noncentral_t.jl")
include("westfall_functions.jl")
# include("variance_functions.jl") # <- TODO

end
