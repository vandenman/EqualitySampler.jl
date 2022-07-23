# Distributions over Partitions

The following distributions are implemented.
```@docs
AbstractMvUrnDistribution
UniformMvUrnDistribution
BetaBinomialMvUrnDistribution
CustomInclusionMvUrnDistribution
RandomProcessMvUrnDistribution
```

Aside from the interface for multivariate distributions, the following methods are also defined.
```@docs
pdf_model,
logpdf_model
pdf_model_distinct
logpdf_model_distinct
pdf_incl
logpdf_incl
```