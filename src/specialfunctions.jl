# convenience wrappers around SpecialFunctions
logbinomial(n::Integer, k::Integer) = SpecialFunctions.logabsbinomial(n, k)[1]
logabsgamma(x::Real) = SpecialFunctions.logabsgamma(x)[1]