# EqualitySampler

[*EqualitySampler*](https://github.com/vandenman/EqualitySampler.jl) defines four distributions over [partitions of a set](https://en.wikipedia.org/wiki/Partition_of_a_set):
- `UniformMvUrnDistribution`
- `BetaBinomialMvUrnDistribution`
- `RandomProcessMvUrnDistribution`
- `CustomInclusionMvUrnDistribution`

Each of these is a subtype of the abstract type `AbstractMvUrnDistribution` which is a subtype of [`Distributions.DiscreteMultivariateDistribution`](https://juliastats.org/Distributions.jl/stable/multivariate/#multivariates).