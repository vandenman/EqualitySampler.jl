# EqualitySampler

[*EqualitySampler*](https://github.com/vandenman/EqualitySampler.jl) defines four distributions over [partitions of a set](https://en.wikipedia.org/wiki/Partition_of_a_set):
- [`UniformMvUrnDistribution`](@ref)
- [`BetaBinomialMvUrnDistribution`](@ref)
- [`CustomInclusionMvUrnDistribution`](@ref)
- [`RandomProcessMvUrnDistribution`](@ref)

Each of these is a subtype of the abstract type `AbstractMvUrnDistribution` which is a subtype of [`Distributions.DiscreteMultivariateDistribution`](https://juliastats.org/Distributions.jl/stable/multivariate/#multivariates).

Thus, each of these types can be called with e.g., `rand` and `logpdf`.

While a partition is usually defined without duplicates, the methods here do assume duplicates are present.
For example, given 3 parameters ``(\theta_1, \theta_2, \theta_3)`` there exist 5 unique partitions:

| partition                                        | constraints                               | representation in Julia |
| ------------------------------------------------ | ----------------------------------------- | ----------------------- |
| ``\{\{\theta_1, \theta_2, \theta_3\}\}``         | ``\theta_1 = \theta_2 = \theta_3``        | `[1, 1, 1]`             |
| ``\{\{\theta_1, \theta_2\}, \{\theta_3\}\}``     | ``\theta_1 = \theta_2 \neq \theta_3``     | `[1, 1, 2]`             |
| ``\{\{\theta_1, \theta_3\}, \{\theta_2\}\}``     | ``\theta_1 = \theta_3 \neq \theta_3``     | `[1, 2, 1]`             |
| ``\{\{\theta_1\}, \{\theta_2, \theta_3\}\}``     | ``\theta_1 \neq \theta_2 = \theta_3``     | `[1, 2, 2]`             |
| ``\{\{\theta_1\}, \{\theta_2\}, \{\theta_3\}\}`` | ``\theta_1 \neq \theta_2 \neq \theta_3``  | `[1, 2, 3]`             |

However, we also consider `[2, 2, 2]` and `[3, 3, 3]` to be valid and identical to `[1, 1, 1]`.
The main reason for this is that in a Gibbs sampling scheme, moving from `[1, 2, 2]` to `[1, 1, 1]` would be a natural transition, however, without duplicated partitions it would also be impossible. The default [`logpdf`](@ref) accounts for duplicated partitions, use [`logpdf_model_distinct`](@ref) to evaluate the logpdf without duplicated partitions.