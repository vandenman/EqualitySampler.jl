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
using EqualitySampler, EqualitySampler.Simulations, Distributions, Statistics
import Random, AbstractMCMC, MCMCChains

# simulate some data
Random.seed!(42) # on julia 1.7.3
n_groups = 5 # no. binomials.
true_partition     = rand(UniformPartitionDistribution(n_groups))
temp_probabilities = rand(n_groups)
true_probabilities = average_equality_constraints(temp_probabilities, true_partition)
# total no. trials
observations = rand(100:200, n_groups)
# no. successes
successes = rand(product_distribution(Binomial.(observations, true_probabilities)))

obs_proportions = successes ./ observations
[true_probabilities obs_proportions]
# 5×2 Matrix{Float64}:
#  0.30743   0.301205
#  0.640909  0.640244
#  0.640909  0.701493
#  0.30743   0.275591
#  0.30743   0.296053

# specify no mcmc iterations, no chains, parallelization. burnin and thinning can also be specified.
mcmc_settings = MCMCSettings(;iterations = 15_000, chains = 4, parallel = AbstractMCMC.MCMCThreads)

# nothing indicates no equality sampling is done and instead the full model is sampled from
chn_full = proportion_test(successes, observations, nothing; mcmc_settings = mcmc_settings)
# use a BetaBinomial(1, k) over the partitions
partition_prior = BetaBinomialPartitionDistribution(n_groups, 1, n_groups)
chn_eqs  = proportion_test(successes, observations, partition_prior; mcmc_settings = mcmc_settings)

# extract posterior mean of full model and averaged across equality constraints
estimated_probabilities_full = mean(chn_full).nt.mean
estimated_probabilities_eqs = mean(MCMCChains.group(chn_eqs, :p_constrained)).nt.mean
[true_probabilities obs_proportions estimated_probabilities_full estimated_probabilities_eqs]
# 5×4 Matrix{Float64}:
#  0.30743   0.301205  0.303421  0.296359
#  0.640909  0.640244  0.638429  0.662943
#  0.640909  0.701493  0.698563  0.666477
#  0.30743   0.275591  0.278896  0.295154
#  0.30743   0.296053  0.298635  0.296189

# posterior probabilities of equalities among the probabilities
compute_post_prob_eq(chn_eqs)
# 5×5 Matrix{Float64}:
#  0.0       0.0      0.0  0.0       0.0
#  0.0       0.0      0.0  0.0       0.0
#  0.0       0.94185  0.0  0.0       0.0
#  0.930683  0.0      0.0  0.0       0.0
#  0.937     0.0      0.0  0.931667  0.0
# true matrix
[i > j && p == q for (i, p) in enumerate(true_partition), (j, q) in enumerate(true_partition)]
# 5×5 Matrix{Bool}:
#  0  0  0  0  0
#  0  0  0  0  0
#  0  1  0  0  0
#  1  0  0  0  0
#  1  0  0  1  0

# list the visited models (use compute_model_probs to obtain their posterior probability)
compute_model_counts(chn_eqs, false)
# OrderedCollections.OrderedDict{String, Int64} with 10 entries:
# "12211" => 51488
# "12215" => 1512
# "12241" => 1808
# "12244" => 1562
# "12245" => 141
# "12311" => 2596
# "12315" => 245
# "12341" => 328
# "12344" => 254
# "12345" => 66

# convert true partition to normalized form and print as string
join(EqualitySampler.reduce_model(true_partition))
# "12211"
# and it so happens to be that the most visited model is also the true model
```
## Post hoc tests in One-Way ANOVA
`anova_test` can be used to explore equality constraints across the levels of a single categorical predictor.
The setup uses a grand mean $\mu$ and offsets $\theta_i$ for every level of the categorical predictor.
To identify the model, the constraint $\sum_i\theta_i = 1$ is imposed.
```julia
using EqualitySampler, EqualitySampler.Simulations, Distributions, Statistics
import Random, AbstractMCMC, MCMCChains

# Simulate some data
Random.seed!(42)
n_groups = 5
n_obs_per_group = 1000
true_partition = rand(UniformPartitionDistribution(n_groups))
temp_θ = randn(n_groups)
temp_θ .-= mean(temp_θ) # center temp_θ to avoid identification constraints
true_θ = average_equality_constraints(temp_θ, true_partition)

g = repeat(1:n_groups; inner = n_obs_per_group)
μ, σ = 0.0, 1.0

# Important: this is the same parametrization as used by the model!
Dy = MvNormal(μ .+ σ .* true_θ[g], σ)
y = rand(Dy)

# observed cell offsets
obs_offset = ([mean(y[g .== i]) for i in unique(g)] .- mean(y)) / std(y)
[true_θ obs_offset]
# 5×2 Matrix{Float64}:
#   0.191185   0.24118
#  -0.286777  -0.290348
#  -0.286777  -0.243936
#   0.191185   0.142092
#   0.191185   0.151012

# specify no mcmc iterations, no chains, parallelization. burnin and thinning can also be specified.
mcmc_settings = MCMCSettings(;iterations = 15_000, chains = 4, parallel = AbstractMCMC.MCMCThreads)

# nothing indicates no equality sampling is done and instead the full model is sampled from
chn_full = anova_test(y, g, nothing; mcmc_settings = mcmc_settings)
# use a BetaBinomial(1, k) over the partitions
partition_prior = BetaBinomialPartitionDistribution(n_groups, 1, n_groups)
chn_eqs  = anova_test(y, g, partition_prior; mcmc_settings = mcmc_settings)

estimated_θ_full = Statistics.mean(MCMCChains.group(chn_full, :θ_cs)).nt.mean
estimated_θ_eqs  = Statistics.mean(MCMCChains.group(chn_eqs , :θ_cs)).nt.mean
[true_θ obs_offset estimated_θ_full estimated_θ_eqs]
# 5×4 Matrix{Float64}:
#   0.191185   0.24118    0.245577   0.194745
#  -0.286777  -0.290348  -0.296143  -0.252687
#  -0.286777  -0.243936  -0.248339  -0.25913
#   0.191185   0.142092   0.145534   0.165273
#   0.191185   0.151012   0.153371   0.1518

# posterior probabilities of equalities among the probabilities
compute_post_prob_eq(chn_eqs)
# 5×5 Matrix{Float64}:
#  0.0         0.0        0.0        0.0     0.0
#  0.00858333  0.0        0.0        0.0     0.0
#  0.0         0.874517   0.0        0.0     0.0
#  0.772967    0.0256833  0.0428167  0.0     0.0
#  0.73465     0.10215    0.0279333  0.8664  0.0
# true matrix
[i > j && p == q for (i, p) in enumerate(true_partition), (j, q) in enumerate(true_partition)]
# 5×5 Matrix{Bool}:
#  0  0  0  0  0
#  0  0  0  0  0
#  0  1  0  0  0
#  1  0  0  0  0
#  1  0  0  1  0

# list the visited models (use compute_model_probs to obtain their posterior probability)
compute_model_counts(chn_eqs, false)
# OrderedCollections.OrderedDict{String, Int64} with 21 entries:
#   "11311" => 421
#   "11341" => 94
#   "12211" => 41409
#   "12212" => 346
#   "12215" => 1161
#   "12221" => 3
#   "12222" => 949
#   "12241" => 427
#   "12242" => 381
#   "12244" => 7699
#   "12245" => 96
#   "12311" => 723
#   "12312" => 2261
#   "12315" => 57
#   "12322" => 168
#   "12331" => 963
#   "12335" => 654
#   "12341" => 39
#   "12342" => 1509
#   "12344" => 615
#   "12345" => 25

# note that there is more uncertainty in the results here, probably because this model is more compelex than the previous.

# convert true partition to normalized form and print as string
join(EqualitySampler.reduce_model(true_partition))
# "12211"
# and it so happens to be that the most visited model is also the true model
```

# Supplementary Analyses
The simulations and analyses for the manuscript 'Flexible Bayesian Multiple Comparison Adjustment Using Dirichlet Process and Beta-Binomial Model Priors' are in the folder "simulations".
Note that this folder is a Julia project, so in order to rerun the simulations it is necessary to first activate and instantiate the project.
