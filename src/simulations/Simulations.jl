module Simulations

using ..EqualitySampler

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
	DataFrames,
	Distributions,
	DistributionsAD,
	DynamicPPL,
	FillArrays,
	GLM,
	KernelDensity,
	LogExpFunctions,
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
	anova_test,
	westfall_test,
	proportion_test,
	compute_post_prob_eq,
	compute_model_probs,
	compute_model_counts

include("partition_sampler.jl")
include("MCMCSettings.jl")
include("brute_force_epsilon.jl")
include("turing_helpers.jl")
include("proportion_functions.jl")
include("simpledataset.jl")
include("anova_functions.jl")
include("logpdf_noncentral_t.jl")
include("westfall_functions.jl")
include("kernel_density.jl")

end
