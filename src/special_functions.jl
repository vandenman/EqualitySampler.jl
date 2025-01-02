# convenience wrappers around SpecialFunctions
logbinomial(n::Integer, k::Integer) = @inbounds SpecialFunctions.logabsbinomial(n, k)[1]