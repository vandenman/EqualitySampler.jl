# EqualitySampler.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://vandenman.github.io/EqualitySampler.jl/dev/)
[![Build Status](https://github.com/vandenman/EqualitySampler/workflows/runtests/badge.svg)](https://github.com/vandenman/EqualitySampler.jl/actions)

EqualitySampler.jl is a Julia library for considering all possible equality constraints across parameters and sampling fromt the posterior distribution over equality constraints.

# Installation

`EqualitySampler` is a registered package, so it can be installed with the usual incantations:
```julia
julia> ]add EqualitySampler
```
or alternatively
```julia
julia> import Pkg; Pkg.add("EqualitySampler")
```

# Functionality

*EqualitySampler* defines four distributions over [partitions of a set](https://en.wikipedia.org/wiki/Partition_of_a_set):
- `UniformPartitionDistribution`
- `BetaBinomialPartitionDistribution`
- `CustomInclusionPartitionDistribution`
- `RandomProcessPartitionDistribution`

Each of these is a subtype of the abstract type `AbstractPartitionDistribution`, which is a subtype of [`Distributions.DiscreteMultivariateDistribution`](https://juliastats.org/Distributions.jl/stable/multivariate/#multivariates).

Thus, each of these types can be called with e.g., `rand` and `logpdf`.

While a partition is usually defined without duplicates, the methods here do assume duplicates are present. For example, given 3 parameters `(θ₁, θ₂, θ₃)` there exist 5 unique partitions:

| partition            | constraints       | representation |
| -------------------- | ----------------- | -------------- |
| `{{θ₁, θ₂, θ₃}}`     | `θ₁ == θ₂ == θ₃`  | `[1, 1, 1]`    |
| `{{θ₁, θ₂}, {θ₃}}`   | `θ₁ == θ₂ != θ₃`  | `[1, 1, 2]`    |
| `{{θ₁, θ₃}, {θ₂}}`   | `θ₁ == θ₃ != θ₂`  | `[1, 2, 1]`    |
| `{{θ₁}, {θ₂, θ₃}}`   | `θ₁ != θ₂ == θ₃`  | `[1, 2, 2]`    |
| `{{θ₁}, {θ₂}, {θ₃}}` | `θ₁ != θ₂ != θ₃`  | `[1, 2, 3]`    |

However, we also consider `[2, 2, 2]` and `[3, 3, 3]` to be valid and identical to `[1, 1, 1]`.
The main reason for this is that in a Gibbs sampling scheme, a transition from `[1, 2, 2]` to `[1, 1, 1]` by updating only the first element would be a natural but impossible without duplicated partitions. The default `logpdf` accounts for duplicated partitions, use `logpdf_model_distinct` to evaluate the logpdf without duplicated partitions.

# Built-in tests

The package contains two functions to explore equality constraints in specific models.
Both use [Turing.jl](github.com/TuringLang/Turing.jl) under the hood and return a Chains object with posterior samples from [MCMCChains.jl](github.com/TuringLang/MCMCChains.jl).

## Independent Binomials
`proportion_test` can be used to explore equality constraints across a series of independent Binomials.
```julia
using EqualitySampler, Distributions
import Random

Random.seed!(42)
n_groups = 5 # no. binomials.
true_partition     = rand(UniformPartitionDistribution(n_groups))
# [1, 2, 2, 1, 1]
temp_probabilities = rand(n_groups)
true_probabilities = temp_probabilities[true_partition]
# [0.165894, 0.613478, 0.613478, 0.165894, 0.165894]

# total no. trials
observations = rand(100:200, n_groups)
# [166, 164, 134, 127, 152]

# no. successes
successes = rand(product_distribution(Binomial.(observations, true_probabilities)))
# [27, 101, 90, 16, 23]

obs_proportions = successes ./ observations
[true_probabilities obs_proportions]
# 5×2 Matrix{Float64}:
#  0.165894  0.162651
#  0.613478  0.615854
#  0.613478  0.671642
#  0.165894  0.125984
#  0.165894  0.151316

partition_prior = BetaBinomialPartitionDistribution(n_groups, 1, n_groups)

no_iter = 100_000

alg_full = EqualitySampler.SampleRJMCMC(;iter = no_iter, fullmodel_only = true)
alg_eq   = EqualitySampler.EnumerateThenSample(;iter = no_iter)

prop_samples_full   = proportion_test(observations, successes, alg_full, partition_prior)
prop_samples_eqs    = proportion_test(observations, successes, alg_eq,   partition_prior)


estimated_probabilities_full = vec(mean(prop_samples_full.parameter_samples.θ_p_samples, dims = 2))
estimated_probabilities_eqs  = vec(mean(prop_samples_eqs.parameter_samples.θ_p_samples, dims = 2))
[true_probabilities obs_proportions estimated_probabilities_full estimated_probabilities_eqs]
# 5×4 Matrix{Float64}:
#  0.165894  0.162651  0.166634  0.150795
#  0.613478  0.615854  0.614287  0.638377
#  0.613478  0.671642  0.669006  0.641879
#  0.165894  0.125984  0.131644  0.149105
#  0.165894  0.151316  0.155732  0.150185

# note the shrinkage towards the group mean. Row 4 has a small observed proportion.
# Since rows 1 and 5 have a high posterior probability of being equal to row 4,

compute_post_prob_eq(prop_samples_eqs)
# 5×5 Matrix{Float64}:
#  0.0          0.0          0.0          0.0     0.0
#  3.33618e-18  0.0          0.0          0.0     0.0
#  3.09553e-20  0.93484      0.0          0.0     0.0
#  0.934021     1.05727e-18  1.0493e-20   0.0     0.0
#  0.944021     1.88964e-18  1.78492e-20  0.9391  0.0

# true equalities
[true_partition[i] == true_partition[j] && i < j for j in eachindex(true_partition), i in eachindex(true_partition)]
# 5×5 Matrix{Bool}:
#  0  0  0  0  0
#  0  0  0  0  0
#  0  1  0  0  0
#  1  0  0  0  0
#  1  0  0  1  0

compute_model_counts(prop_samples_eqs, false)
# OrderedCollections.OrderedDict{Vector{Int8}, Int64} with 10 entries:
#   [1, 2, 2, 1, 1] => 86102
#   [1, 2, 3, 1, 1] => 4938
#   [1, 2, 2, 3, 1] => 2777
#   [1, 2, 2, 3, 3] => 2466
#   [1, 2, 2, 1, 3] => 1993
#   [1, 2, 3, 4, 1] => 542
#   [1, 2, 3, 4, 4] => 442
#   [1, 2, 3, 1, 4] => 370
#   [1, 2, 2, 3, 4] => 264
#   [1, 2, 3, 4, 5] => 106

# and it so happens to be that the most visited model is also the true model
true_partition
# 5-element Vector{Int64}:
#  1
#  2
#  2
#  1
#  1
```
## Post hoc tests in One-Way ANOVA
`anova_test` can be used to explore equality constraints across the levels of a single categorical predictor.
The setup uses a grand mean $\mu$ and offsets $\theta_i$ for every level of the categorical predictor.
To identify the model, the constraint $\sum_i\theta_i = 1$ is imposed.
```julia
using EqualitySampler, Distributions
import Random

# Simulate some data
Random.seed!(31415)
n_groups = 7
n_obs_per_group = 50
true_partition = rand(UniformPartitionDistribution(n_groups))

temp_θ = randn(n_groups)
true_θ = temp_θ[true_partition]
true_θ .-= mean(true_θ) # center temp_θ to avoid identification constraints

data_obj = simulate_data_one_way_anova(n_groups, n_obs_per_group, true_θ, true_partition)
data = data_obj.data

obs_offset = ([mean(data.y[data.g[i]]) for i in eachindex(data.g)] .- mean(data.y)) / std(data.y)
[true_θ obs_offset]
# 7×2 Matrix{Float64}:
#   0.253441   0.289433
#   0.253441   0.231883
#   0.253441   0.0709005
#  -0.197943  -0.297703
#   0.253441   0.339342
#   0.204135   0.277509
#  -1.01996   -0.911364

iter = 100_000

alg_full = EqualitySampler.SampleRJMCMC(;iter = no_iter, fullmodel_only = true)
alg_eq   = EqualitySampler.EnumerateThenSample(;iter = no_iter)

partition_prior = BetaBinomialPartitionDistribution(n_groups, 1, n_groups)
anova_samples_full   = anova_test(data.y, data.g, alg_full, partition_prior)
anova_samples_eqs    = anova_test(data.y, data.g, alg_eq,   partition_prior)

estimated_θ_full = vec(mean(anova_samples_full.parameter_samples.θ_cp, dims = 2))
estimated_θ_eqs  = vec(mean( anova_samples_eqs.parameter_samples.θ_cp, dims = 2))
[true_θ obs_offset estimated_θ_full estimated_θ_eqs]
# 7×4 Matrix{Float64}:
#   0.253441   0.289433    0.287095    0.250173
#   0.253441   0.231883    0.23036     0.237068
#   0.253441   0.0709005   0.0710826   0.155485
#  -0.197943  -0.297703   -0.296033   -0.280944
#   0.253441   0.339342    0.337171    0.258582
#   0.204135   0.277509    0.275538    0.247956
#  -1.01996   -0.911364   -0.905213   -0.86832


# posterior probabilities of equalities among the probabilities
compute_post_prob_eq(anova_samples_eqs)
# 7×7 Matrix{Float64}:
#  0.0         0.0         0.0         0.0        0.0         0.0         0.0
#  0.73923     0.0         0.0         0.0        0.0         0.0         0.0
#  0.591839    0.618516    0.0         0.0        0.0         0.0         0.0
#  0.0366705   0.0481045   0.177259    0.0        0.0         0.0         0.0
#  0.762704    0.736683    0.57209     0.0316826  0.0         0.0         0.0
#  0.753677    0.739104    0.597044    0.0384047  0.75827     0.0         0.0
#  5.46569e-8  2.93418e-7  5.64801e-5  0.129007   1.40949e-8  7.66761e-8  0.0

# true matrix
[i > j && p == q for (i, p) in enumerate(true_partition), (j, q) in enumerate(true_partition)]
# 7×7 Matrix{Bool}:
#  0  0  0  0  0  0  0
#  1  0  0  0  0  0  0
#  1  1  0  0  0  0  0
#  0  0  0  0  0  0  0
#  1  1  1  0  0  0  0
#  0  0  0  0  0  0  0
#  0  0  0  0  0  0  0

# note that the model mistakenly identifies groups 5 and 6 as equal
# these had 0.253441 and 0.339342 as their sample values and 0.204135 and 0.277509 as their true means.

# list the visited models (use compute_model_probs to obtain their posterior probability)
compute_model_counts(anova_samples_eqs, false)
# OrderedCollections.OrderedDict{Vector{Int8}, Int64} with 242 entries:
#   [1, 1, 1, 2, 1, 1, 3] => 32242
#   [1, 1, 1, 2, 1, 1, 2] => 10256
#   [1, 1, 2, 2, 1, 1, 3] => 9223
#   [1, 1, 2, 3, 1, 1, 4] => 4421
#   [1, 2, 2, 3, 1, 1, 4] => 2511
#   [1, 1, 1, 1, 1, 1, 2] => 2469
#   [1, 2, 2, 3, 1, 2, 4] => 1807
#   [1, 1, 2, 3, 1, 2, 4] => 1784
#   [1, 1, 1, 2, 3, 1, 4] => 1772
#   [1, 1, 1, 2, 3, 3, 4] => 1666
#   [1, 2, 1, 3, 2, 2, 4] => 1637
#   [1, 2, 2, 3, 2, 2, 4] => 1419
#   ⋮                     => ⋮

```

# Supplementary Analyses
The simulations and analyses for the manuscript 'Flexible Bayesian Multiple Comparison Adjustment Using Dirichlet Process and Beta-Binomial Model Priors' are in the folder "simulations".
Note that this folder is a Julia project, so in order to rerun the simulations it is necessary to first activate and instantiate the project.
