# convenience wrappers around SpecialFunctions
logbinomial(n::Integer, k::Integer) = @inbounds SpecialFunctions.logabsbinomial(n, k)[1]
logabsgamma(x::Real) = @inbounds SpecialFunctions.logabsgamma(x)[1]