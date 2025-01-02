module EqualitySampler

# stdlib
import Base: length
import LinearAlgebra, Random, Statistics

import
    Combinatorics,
    DataFrames,                 # keep/ extension?
    Distributions,              # keep
    FillArrays,                 # keep
    HypergeometricFunctions,    # keep
    IrrationalConstants,        # keep
    LogarithmicNumbers,         # keep / extension
    LogExpFunctions,            # keep
    OrderedCollections,         # keep
    Optim,                      # keep
    PDMats,                     # keep
    ProgressMeter,              # keep
    StatsBase,                  # keep
    StatsModels,                # keep / extension
    SpecialFunctions,           # keep
    Suppressor,                 # need for now
    QuadGK                      # keep

using DocStringExtensions
using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF, TYPEDFIELDS

export

    # combinatorial functions
    stirlings2,
    stirlings2r,
    logstirlings2,
    logstirlings2r,

    unsignedstirlings1,
    logunsignedstirlings1,

    # TODO: export the generalized stirling function?

    bellnum,
    bellnumr,
    logbellnumr,

    # simple counting functions
    count_equalities,
    count_parameters,

    PartitionSpace,

    AbstractPartitionDistribution,
    AbstractSizePartitionDistribution,
    AbstractProcessPartitionDistribution,
    UniformPartitionDistribution,
    BetaBinomialPartitionDistribution,
    CustomInclusionPartitionDistribution,
    DuplicatedPartitionDistribution,
    PrecomputedCustomInclusionPartitionDistribution,
    RandomProcessPartitionDistribution,
    DirichletProcessPartitionDistribution,
    PitmanYorProcessPartitionDistribution,

    pdf_model,
    pdf_model_distinct,
    pdf_incl,

    logpdf_model,
    logpdf_model_distinct,
    logpdf_incl,

    prediction_rule,
    tie_probability,

    # remove everything below these
    get_normal_dense_chol_suff_stats,
    logpdf_mv_normal_chol_suffstat,
    loglikelihood_suffstats,

    # Jeffreys priors for variances and sds
    JeffreysPriorStandardDeviation,
    JeffreysPriorVariance,


    # tests
    anova_test,
    proportion_test,
    westfall_test,
    proportion_test,

    # simulate data
    simulate_data_one_way_anova,
    simulate_proportions,

    # postprocessing helpers
    compute_post_prob_eq,
    compute_model_probs,
    compute_model_counts,
    compute_model_probs2,

    get_hpm_partition,
    get_mpm_partition



include("noncentral_t.jl")
include("special_functions.jl")
include("stirling_helpers.jl")
include("stirling1.jl")
include("stirling2.jl")
include("rstirling2.jl")
include("generalized_stirling.jl")
include("bellnumr.jl")
include("partition_shortcuts.jl")
include("partitionspace.jl")
include("combinatorics.jl")
include("abstractpartitiondistribution.jl")
include("abstractsizepartitiondistribution.jl")
include("abstractprocesspartitiondistribution.jl")
include("uniformpartitiondistribution.jl")
include("betabinomialpartitiondistribution.jl")
include("custominclusionpartitiondistribution.jl")
include("duplicatedpartitiondistribution.jl")
include("pitman_yorpartitiondistribution.jl")
include("conditional_urn_distributions.jl")
include("normal_loglikelihood.jl")
include("helpers.jl")
include("dpp_find_alpha.jl")

# before in simulations
include("simpledataset.jl")				# could just be a namedtuple? or a DataFrame?
include("abstractsamplingmethod.jl")
include("split_merge_move.jl")
include("anova_functions.jl")
include("proportion_functions.jl")
include("westfall_functions.jl")
include("find_initial_partition.jl")
include("postprocess_samples.jl")


# include("simulations/Simulations.jl")
# import .Simulations: anova_test, proportion_test,
# 	compute_model_probs2,
# 	simulate_proportions,
# 	proportions_sample,
# 	proportions_sample_integrated,
# 	proportions_enumerate

# export
# 	anova_test,
# 	proportion_test,
# 	average_equality_constraints,
# 	MCMCSettings,

# 	# anova_sample, anova_sample_integrated, anova_enumerate,
# 	compute_model_probs2,
# 	simulate_proportions,
# 	proportions_sample,
# 	proportions_sample_integrated,
# 	proportions_enumerate



end

