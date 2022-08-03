# EqualitySampler.jl - a Julia package for sampling equality constraints

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://vandenman.github.io/EqualitySampler/dev/)
[![Build Status](https://github.com/vandenman/EqualitySampler/workflows/runtests/badge.svg)](https://github.com/vandenman/EqualitySampler/EqualitySampler/actions)

# Installation

## TODO: Actually register the package!

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
The main reason for this is that in a Gibbs sampling scheme, a transition from `[1, 2, 2]` to `[1, 1, 1]` would be a natural but impossible without duplicated partitions. The default `logpdf` accounts for duplicated partitions, use `logpdf_model_distinct` to evaluate the logpdf without duplicated partitions.

# Built in tests

`anova_test`

`proportion_test`

```julia
using EqualitySampler, Distributions
n_groups  = 5
true_partition     = rand(UniformPartitionDistribution(n_groups))
true_probabilities = average_equality_constraints
obs_counts    = rand(100:200, n_groups)
obs_successes =
```

# Use in Turing models

The distributions over partitions can be used in Turing models.
For example, one could do
```julia
@model function hotellingsT(x)

    p = size(x, 2)
    # partition prior
    partition ~ BetaBinomialPartitionDistrbiution(p, 1, p)

    # right Haar prior on the grand mean and standard deviation
    μ ~ Turing.Flat()
    logσ ~ Turing.Flat()

    σ = exp(logσ)

    # prior on unconstrained offsets
    θ ~ MvNormal(p, 5)
    # constrains offsets according to
    θc = EqualitySampler.average_equality_constraints(θ .- mean(θ), partition)

    # Likelihood
    x ~ MvNormal(θc, σ)

end
```
Note: there is a better way than `θ .- mean(θ)`!