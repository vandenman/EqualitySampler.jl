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

		- [ ] write and export one function that computes the likelihood using Distributions.suffstats(MvNormal, x) and Distributions.suffstats(Normal, x)
			perhaps skip/ remove logpdf_mv_normal_chol_suffstat and friends (constructing the full matrix is not super expensive)
			also use PDMats.invquad and friends

		- [ ] consider renaming "model" to "partition" everywhere. For example,
			AbstractMvUrnDistribution	-> AbstractPartitionDistribution
			reduce_model 				-> normalize_partition



	=#

	get_normal_dense_chol_suff_stats,
	logpdf_mv_normal_chol_suffstat,
	loglikelihood_suffstats,

	# combinatorial functions
	stirlings2,
	stirlings2r,
	logstirlings2,
	logstirlings2r,

	unsignedstirlings1,
	logunsignedstirlings1,

	bellnum,
	bellnumr,
	logbellnumr,

	# simple counting functions
	count_equalities,
	count_parameters,

	# TODO: deprecate in favor of the iterator
	PartitionSpace,

	AbstractMvUrnDistribution,
	UniformMvUrnDistribution,
	BetaBinomialMvUrnDistribution,
	CustomInclusionMvUrnDistribution,
	RandomProcessMvUrnDistribution,
	DirichletProcessMvUrnDistribution,

	pdf_model,
	pdf_model_distinct,
	pdf_incl,

	logpdf_model,
	logpdf_model_distinct,
	logpdf_incl,

	# Jeffreys priors for variances and sds
	JeffreysPriorStandardDeviation,
	JeffreysPriorVariance


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

