module Simulations

using ..EqualitySampler

# TODO: should we just do using Turing?

# stdlib
import
	LinearAlgebra,
	Logging

import
	Statistics: mean, var

# other dependencies
import
	CategoricalArrays,
	DataFrames,
	Distributions,
	DistributionsAD,
	DynamicPPL,
	FillArrays,
	GLM,
	MCMCChains,
	OrderedCollections,
	StatsBase,
	StatsModels,
	Turing

export
	anova_test,
	proportion_test#,
	# variance_test

include("proportion_functions.jl")
include("anova_functions.jl")
# include("variance_functions.jl") # <- TODO

end

# module Atest
# 	export f
# 	f(x) = x+1
# 	module Btest
# 		# import ..Atest
# 		# g(x) = Atest.f(x)
# 		using ..Atest
# 		g(x) = f(x)
# 	end
# end