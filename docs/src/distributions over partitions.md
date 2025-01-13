# Distributions over Partitions

The following distributions are implemented.
```@docs
AbstractPartitionDistribution
UniformPartitionDistribution
BetaBinomialPartitionDistribution
CustomInclusionPartitionDistribution
DirichletProcessPartitionDistribution
PitmanYorProcessPartitionDistribution
PrecomputedCustomInclusionPartitionDistribution
DuplicatedPartitionDistribution
```

Aside from the interface for multivariate distributions, the following methods are also defined.
```@docs
pdf_model
logpdf_model
pdf_model_distinct
logpdf_model_distinct
pdf_incl
logpdf_incl
```