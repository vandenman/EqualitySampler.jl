# EqualitySampler

[*EqualitySampler*](https://github.com/vandenman/EqualitySampler.jl) defines distributions over [partitions of a set](https://en.wikipedia.org/wiki/Partition_of_a_set):
- [`UniformPartitionDistribution`](@ref)
- [`BetaBinomialPartitionDistribution`](@ref)
- [`CustomInclusionPartitionDistribution`](@ref)
- [`DirichletProcessPartitionDistribution`](@ref)
- [`PitmanYorProcessPartitionDistribution`](@ref)

These distributions can be as prior distributions over equality constraints among a set of variables.

## Type Hierarchy

Each of the distributions is a subtype of `AbstractPartitionDistribution` which is a subtype of [`Distributions.DiscreteMultivariateDistribution`](https://juliastats.org/Distributions.jl/stable/multivariate/#multivariates).

Thus, each of these types can be called with e.g., `rand` and `logpdf`.
There are multiple abstract types, in particular there is `AbstractSizePartitionDistribution` which is a subtype of `AbstractPartitionDistribution` and is used to represent distributions over partitions where the logpdf of a partition is entirely determined by the size of partition (e.g., ``\{\{\theta_1, \theta_2\}, \{\theta_3\}\}`` has size 2).
When using subtypes of this distribution, it may be convenient to use `PrecomputedCustomInclusionPartitionDistribution` which caches all the computations.
There is also `AbstractProcessPartitionDistribution` for distributions that are based on a stochastic process, such as the Pitman-Yor process and the Dirichlet Process.

## Representation of Partitions

While a partition is usually defined without duplicates, the methods here do assume duplicates are present.
For example, given 3 parameters ``(\theta_1, \theta_2, \theta_3)`` there exist 5 unique partitions:

| partition                                        | constraints                               | representation in Julia |
| ------------------------------------------------ | ----------------------------------------- | ----------------------- |
| ``\{\{\theta_1, \theta_2, \theta_3\}\}``         | ``\theta_1 = \theta_2 = \theta_3``        | `[1, 1, 1]`             |
| ``\{\{\theta_1, \theta_2\}, \{\theta_3\}\}``     | ``\theta_1 = \theta_2 \neq \theta_3``     | `[1, 1, 2]`             |
| ``\{\{\theta_1, \theta_3\}, \{\theta_2\}\}``     | ``\theta_1 = \theta_3 \neq \theta_3``     | `[1, 2, 1]`             |
| ``\{\{\theta_1\}, \{\theta_2, \theta_3\}\}``     | ``\theta_1 \neq \theta_2 = \theta_3``     | `[1, 2, 2]`             |
| ``\{\{\theta_1\}, \{\theta_2\}, \{\theta_3\}\}`` | ``\theta_1 \neq \theta_2 \neq \theta_3``  | `[1, 2, 3]`             |

We do not consider `[2, 2, 2]` and `[3, 3, 3]` to be valid partitions identical to `[1, 1, 1]`.
When this is desired, one can wrap a partition distribution using `DuplicatedPartitionDistribution` to allow duplicated representation for a partition.