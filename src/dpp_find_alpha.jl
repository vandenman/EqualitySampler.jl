
"""
$(TYPEDSIGNATURES)

Computes ``\\alpha`` such that the probability of a model where all variables are the same is equal to that of a model where all variables are distinct.
This does not consider duplicte models, i.e., it minimizes ``(EqualitySampler.logpdf_model_distinct(D, null_model) - EqualitySampler.logpdf_model_distinct(D, full_model))^2`` as a function of ``\\alpha``.
"""
dpp_find_α(k::Integer) = exp(-(1 / (1 - k)) * SpecialFunctions.loggamma(k))
