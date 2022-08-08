var documenterSearchIndex = {"docs":
[{"location":"tests/#Independent-Binomials","page":"Independent Binomials","title":"Independent Binomials","text":"","category":"section"},{"location":"tests/","page":"Independent Binomials","title":"Independent Binomials","text":"proportion_test","category":"page"},{"location":"tests/#EqualitySampler.Simulations.proportion_test","page":"Independent Binomials","title":"EqualitySampler.Simulations.proportion_test","text":"proportion_test(successes::AbstractArray{T<:Integer, 1}, observations::AbstractArray{T<:Integer, 1}, partition_prior::Union{Nothing, AbstractPartitionDistribution}; spl, mcmc_settings, ϵ, n_leapfrog, kwargs...) -> Any\n\n\nFit independent binomials to the successes and observations and explore equality constraints among the probabilities.\n\nArguments\n\nsuccesses, vector of successes.\nobservations vector of no trials.\npartition_prior, the prior to use over partitions or nothing, which implies sampling from the full model.\n\nKeyword arguments\n\nspl, overwrite the sampling algorithm passed to Turing. It's best to look at the source code for the parameter names and so on.\nmcmc_settings, settings for sampling.\nϵ, passed to Turing.HMC, only used when partition_prior !== nothing.\nn_leapfrog, passed to Turing.HMC, only used when partition_prior !== nothing.\nkwargs..., passed to AbstractMCMC.sample.\n\n\n\n\n\n","category":"function"},{"location":"tests/#Post-hoc-tests-in-One-Way-ANOVA","page":"Independent Binomials","title":"Post hoc tests in One-Way ANOVA","text":"","category":"section"},{"location":"tests/","page":"Independent Binomials","title":"Independent Binomials","text":"anova_test","category":"page"},{"location":"tests/#EqualitySampler.Simulations.anova_test","page":"Independent Binomials","title":"EqualitySampler.Simulations.anova_test","text":"anova_test(f::StatsModels.FormulaTerm, df::DataFrames.DataFrame, args...; kwargs...) -> Any\n\n\nUsing the formula f and data frame df fit a one-way ANOVA.\n\n\n\n\n\nanova_test(y::AbstractVector{var\"#s199\"} where var\"#s199\"<:AbstractFloat, g::AbstractVector{var\"#s192\"} where var\"#s192\"<:Integer, args...; kwargs...) -> Any\n\n\nUsing the vector y and grouping variable g fit a one-way ANOVA.\n\n\n\n\n\nanova_test(y::AbstractVector{var\"#s41\"} where var\"#s41\"<:AbstractFloat, g::AbstractVector{var\"#s40\"} where var\"#s40\"<:(UnitRange{var\"#s39\"} where var\"#s39\"<:Integer), args...; kwargs...) -> Any\n\n\nUsing the vector y and grouping variable g fit a one-way ANOVA. Here g is a vector of UnitRanges where each element indicates the group membership of y.\n\n\n\n\n\nanova_test(df::Union{DataFrames.DataFrame, EqualitySampler.Simulations.SimpleDataSet}, partition_prior::Union{Nothing, AbstractPartitionDistribution}; spl, mcmc_settings, modeltype, rng) -> Any\n\n\nArguments:\n\ndf a DataFrame or SimpleDataSet.\npartition_prior::Union{Nothing, AbstractPartitionDistribution}, either nothing (i.e., fit the full model) or a subtype of AbstractPartitionDistribution.\n\nKeyword arguments\n\nspl, overwrite the sampling algorithm passed to Turing. It's best to look at the source code for the parameter names and so on.\nmcmc_settings, settings for sampling.\nmodeltype, :old indicated all parameters are sampled whereas reduced indicates only g and the partitions are sampled using an integrated representation of the posterior.\nrng a random number generator.\n\n\n\n\n\n","category":"function"},{"location":"combinatorial functions/#combinatorics","page":"Combinatorial Functions","title":"Combinatorial Functions","text":"","category":"section"},{"location":"combinatorial functions/","page":"Combinatorial Functions","title":"Combinatorial Functions","text":"stirlings2\nlogstirlings2\nstirlings2r\nlogstirlings2r\nunsignedstirlings1\nlogunsignedstirlings1\nbellnum\nbellnumr\nlogbellnumr\nPartitionSpace","category":"page"},{"location":"combinatorial functions/#EqualitySampler.stirlings2","page":"Combinatorial Functions","title":"EqualitySampler.stirlings2","text":"Compute the Stirlings numbers of the second kind. The EqualitySampler.ExplicitStrategy (default) uses an explicit loop and is computationally more efficient but subject to overflow, so using BigInt is advised. The EqualitySampler.RecursiveStrategy uses recursion and is mathematically elegant yet inefficient for large values.\n\n\n\n\n\n","category":"function"},{"location":"combinatorial functions/#EqualitySampler.logstirlings2","page":"Combinatorial Functions","title":"EqualitySampler.logstirlings2","text":"Compute the logarithm of the Stirlings numbers of the second kind with an explicit formula.\n\n\n\n\n\n","category":"function"},{"location":"combinatorial functions/#EqualitySampler.stirlings2r","page":"Combinatorial Functions","title":"EqualitySampler.stirlings2r","text":"stirlings2r(n::T, k::T, r::T) where T <: Integer\nstirlings2r(n::T, k::T, r::T, ::Type{EqualitySampler.ExplicitStrategy})  where T <: Integer\nstirlings2r(n::T, k::T, r::T, ::Type{EqualitySampler.RecursiveStrategy}) where T <: Integer\n\nCompute the r-Stirlings numbers of the second kind. The EqualitySampler.ExplicitStrategy (default) uses an explicit loop and is computationally more efficient but subject to overflow, so using BigInt is advised. The EqualitySampler.RecursiveStrategy uses recursion and is mathematically elegant yet inefficient for large values.\n\n\n\n\n\n","category":"function"},{"location":"combinatorial functions/#EqualitySampler.logstirlings2r","page":"Combinatorial Functions","title":"EqualitySampler.logstirlings2r","text":"logstirlings2r(n::T, k::T, r::T) where T <: Integer\n\nComputes the logarithm of the r-stirling numbers with an explicit formula.\n\n\n\n\n\n","category":"function"},{"location":"combinatorial functions/#EqualitySampler.unsignedstirlings1","page":"Combinatorial Functions","title":"EqualitySampler.unsignedstirlings1","text":"unsignedstirlings1(n::Integer, k::Integer) -> Any\n\n\nCompute the absolute value of the Stirlings numbers of the first kind. The EqualitySampler.ExplicitStrategy (default) uses an explicit loop and is computationally more efficient but subject to overflow, so using BigInt is advised. The EqualitySampler.RecursiveStrategy uses recursion and is mathematically elegant yet inefficient for large values.\n\n\n\n\n\n","category":"function"},{"location":"combinatorial functions/#EqualitySampler.logunsignedstirlings1","page":"Combinatorial Functions","title":"EqualitySampler.logunsignedstirlings1","text":"logunsignedstirlings1(n::Integer, k::Integer) -> Any\n\n\nCompute the logarithm of the absolute value of the Stirlings numbers of the first kind.\n\n\n\n\n\n","category":"function"},{"location":"combinatorial functions/#EqualitySampler.bellnum","page":"Combinatorial Functions","title":"EqualitySampler.bellnum","text":"Computes the Bell numbers.\n\n\n\n\n\n","category":"function"},{"location":"combinatorial functions/#EqualitySampler.bellnumr","page":"Combinatorial Functions","title":"EqualitySampler.bellnumr","text":"Computes the r-Bell numbers.\n\n\n\n\n\n","category":"function"},{"location":"combinatorial functions/#EqualitySampler.logbellnumr","page":"Combinatorial Functions","title":"EqualitySampler.logbellnumr","text":"Computes the logarithm of the r-Bell numbers.\n\n\n\n\n\n","category":"function"},{"location":"combinatorial functions/#EqualitySampler.PartitionSpace","page":"Combinatorial Functions","title":"EqualitySampler.PartitionSpace","text":"struct PartitionSpace{T<:Integer, P<:EqualitySampler.AbstractPartitionSpace}\n\n# constructor\nPartitionSpace(k::T, ::Type{U} = EqualitySampler.DistinctPartitionSpace)\n\nA type to represent the space of partitions. EqualitySampler.AbstractPartitionSpace indicates whether partitions should contains duplicates or be distinct. For example, the distinct iterator will return [1, 1, 2] but not [2, 2, 1] and [1, 1, 3], which are returned when P = EqualitySampler.DuplicatedPartitionSpace.\n\n\n\n\n\n","category":"type"},{"location":"distributions over partitions/#Distributions-over-Partitions","page":"Distributions over Partitions","title":"Distributions over Partitions","text":"","category":"section"},{"location":"distributions over partitions/","page":"Distributions over Partitions","title":"Distributions over Partitions","text":"The following distributions are implemented.","category":"page"},{"location":"distributions over partitions/","page":"Distributions over Partitions","title":"Distributions over Partitions","text":"AbstractPartitionDistribution\nUniformPartitionDistribution\nBetaBinomialPartitionDistribution\nCustomInclusionPartitionDistribution\nRandomProcessPartitionDistribution","category":"page"},{"location":"distributions over partitions/#EqualitySampler.AbstractPartitionDistribution","page":"Distributions over Partitions","title":"EqualitySampler.AbstractPartitionDistribution","text":"AbstractPartitionDistribution{<:Integer} <: Distributions.DiscreteMultivariateDistribution\n\nSupertype for distributions over partitions.\n\n\n\n\n\n","category":"type"},{"location":"distributions over partitions/#EqualitySampler.UniformPartitionDistribution","page":"Distributions over Partitions","title":"EqualitySampler.UniformPartitionDistribution","text":"UniformPartitionDistribution{T <: Integer} <: AbstractPartitionDistribution{T}\n\nUniform distribution over partitions.\n\n\n\n\n\n","category":"type"},{"location":"distributions over partitions/#EqualitySampler.BetaBinomialPartitionDistribution","page":"Distributions over Partitions","title":"EqualitySampler.BetaBinomialPartitionDistribution","text":"BetaBinomialPartitionDistribution{T <: Integer} <: AbstractPartitionDistribution{T}\n\nBeta binomial distribution over partitions. If rho sim textBetaBinomialPartitionDistribution(k alpha beta) then textcount_parameters(rho) sim textBetaBinomial(k - 1 alpha beta).\n\n\n\n\n\n","category":"type"},{"location":"distributions over partitions/#EqualitySampler.CustomInclusionPartitionDistribution","page":"Distributions over Partitions","title":"EqualitySampler.CustomInclusionPartitionDistribution","text":"CustomInclusionPartitionDistribution(k::T, logpdf::NTuple{N, Float64})\n\nCustomInclusionPartitionDistribution is similar to the BetaBinomialPartitionDistribution in that the model probabilities are completely determined by the size of the partition. Whereas the BetaBinomialPartitionDistribution uses a BetaBinomial distribution to obtain the probabilities, the CustomInclusionPartitionDistribution can be used to specify any vector of probabilities. This distribution is particularly useful to sample uniformly from partitions of a given size. For example:\n\nrand(CustomInclusionPartitionDistribution(4, ntuple(i->log(i==1), Val(4)))) # always all equal (1 parameter)\nrand(CustomInclusionPartitionDistribution(4, ntuple(i->log(i==3), Val(4)))) # always 3 parameters\nrand(CustomInclusionPartitionDistribution(4, ntuple(i->log(i==4), Val(4)))) # always completely distinct (4 parameters)\n\nThe function does not check if sum(exp, logpdf) ≈ 1.0, that is the callers responsibility.\n\n\n\n\n\n","category":"type"},{"location":"distributions over partitions/#EqualitySampler.RandomProcessPartitionDistribution","page":"Distributions over Partitions","title":"EqualitySampler.RandomProcessPartitionDistribution","text":"RandomProcessPartitionDistribution{RPM <: Turing.RandomMeasures.AbstractRandomProbabilityMeasure, T <: Integer} <: AbstractPartitionDistribution{T}\n\nDistribution over partitions defined by a Random Probabiltiy Measure (RPM) as defined in Turing.RandomMeasures.\n\n\n\n\n\n","category":"type"},{"location":"distributions over partitions/","page":"Distributions over Partitions","title":"Distributions over Partitions","text":"Aside from the interface for multivariate distributions, the following methods are also defined.","category":"page"},{"location":"distributions over partitions/","page":"Distributions over Partitions","title":"Distributions over Partitions","text":"pdf_model\nlogpdf_model\npdf_model_distinct\nlogpdf_model_distinct\npdf_incl\nlogpdf_incl","category":"page"},{"location":"distributions over partitions/#EqualitySampler.pdf_model","page":"Distributions over Partitions","title":"EqualitySampler.pdf_model","text":"pdf_model(d::AbstractPartitionDistribution, x::Integer)\npdf_model(d::AbstractPartitionDistribution, x::AbstractVector{<:Integer})\n\nSynonym for pdf(d::AbstractPartitionDistribution, x), computes the probability of a partition.\n\n\n\n\n\n","category":"function"},{"location":"distributions over partitions/#EqualitySampler.logpdf_model","page":"Distributions over Partitions","title":"EqualitySampler.logpdf_model","text":"logpdf_model(d::AbstractPartitionDistribution, x::Integer)\nlogpdf_model(d::AbstractPartitionDistribution, x::AbstractVector{<:Integer})\n\nSynonym for logpdf(d::AbstractPartitionDistribution, x), computes the log probability of a partition.\n\n\n\n\n\n","category":"function"},{"location":"distributions over partitions/#EqualitySampler.pdf_model_distinct","page":"Distributions over Partitions","title":"EqualitySampler.pdf_model_distinct","text":"pdf_model_distinct(d::AbstractPartitionDistribution, x)\n\nComputes the probability of a partition without considering duplicated partitions (i.e., assuming all partitions are unique).\n\n\n\n\n\n","category":"function"},{"location":"distributions over partitions/#EqualitySampler.logpdf_model_distinct","page":"Distributions over Partitions","title":"EqualitySampler.logpdf_model_distinct","text":"logpdf_model_distinct(d::AbstractPartitionDistribution, x)\n\nComputes the log probability of a partition without considering duplicated partitions (i.e., assuming all partitions are unique).\n\n\n\n\n\n","category":"function"},{"location":"distributions over partitions/#EqualitySampler.pdf_incl","page":"Distributions over Partitions","title":"EqualitySampler.pdf_incl","text":"pdf_incl(d::AbstractPartitionDistribution, no_parameters::Integers)\n\nProbability of all partitions with a particular number of parameters.\n\n\n\n\n\n","category":"function"},{"location":"distributions over partitions/#EqualitySampler.logpdf_incl","page":"Distributions over Partitions","title":"EqualitySampler.logpdf_incl","text":"logpdf_incl(d::AbstractPartitionDistribution, no_parameters::Integers)\n\nLog probability of all partitions with a particular number of parameters.\n\n\n\n\n\n","category":"function"},{"location":"#EqualitySampler","page":"EqualitySampler","title":"EqualitySampler","text":"","category":"section"},{"location":"","page":"EqualitySampler","title":"EqualitySampler","text":"EqualitySampler defines four distributions over partitions of a set:","category":"page"},{"location":"","page":"EqualitySampler","title":"EqualitySampler","text":"UniformPartitionDistribution\nBetaBinomialPartitionDistribution\nCustomInclusionPartitionDistribution\nRandomProcessPartitionDistribution","category":"page"},{"location":"","page":"EqualitySampler","title":"EqualitySampler","text":"These distributions can be as prior distributions over equality constraints among a set of variables.","category":"page"},{"location":"#Type-Hierarchy","page":"EqualitySampler","title":"Type Hierarchy","text":"","category":"section"},{"location":"","page":"EqualitySampler","title":"EqualitySampler","text":"Each of the four distributions is a subtype of AbstractPartitionDistribution which is a subtype of Distributions.DiscreteMultivariateDistribution.","category":"page"},{"location":"","page":"EqualitySampler","title":"EqualitySampler","text":"Thus, each of these types can be called with e.g., rand and logpdf.","category":"page"},{"location":"#Representation-of-Partitions","page":"EqualitySampler","title":"Representation of Partitions","text":"","category":"section"},{"location":"","page":"EqualitySampler","title":"EqualitySampler","text":"While a partition is usually defined without duplicates, the methods here do assume duplicates are present. For example, given 3 parameters (theta_1 theta_2 theta_3) there exist 5 unique partitions:","category":"page"},{"location":"","page":"EqualitySampler","title":"EqualitySampler","text":"partition constraints representation in Julia\ntheta_1 theta_2 theta_3 theta_1 = theta_2 = theta_3 [1, 1, 1]\ntheta_1 theta_2 theta_3 theta_1 = theta_2 neq theta_3 [1, 1, 2]\ntheta_1 theta_3 theta_2 theta_1 = theta_3 neq theta_3 [1, 2, 1]\ntheta_1 theta_2 theta_3 theta_1 neq theta_2 = theta_3 [1, 2, 2]\ntheta_1 theta_2 theta_3 theta_1 neq theta_2 neq theta_3 [1, 2, 3]","category":"page"},{"location":"","page":"EqualitySampler","title":"EqualitySampler","text":"However, we also consider [2, 2, 2] and [3, 3, 3] to be valid and identical to [1, 1, 1]. The main reason for this is that it has pragmatic benefits when doing Gibbs sampling. For example, consider that the current partition is [1, 2, 2] and the sampler proposes an update for the first element. A natural proposal would be [1, 1, 1], however, without duplicated partitions this would be impossible (as [2, 2, 2] would not exist). The default logpdf accounts for duplicated partitions, use logpdf_model_distinct to evaluate the logpdf without duplicated partitions.","category":"page"}]
}
