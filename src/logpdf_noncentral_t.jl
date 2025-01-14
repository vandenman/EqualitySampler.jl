# this approach below avoid Rmath which is nice, but sadly it's not accurate enough.

# function Hermite2(n::Integer, x::T) where T<:Number
# 	# generated with mathematica
# 	res = (
# 		-2 * x * 	SpecialFunctions.gamma(0.5 - n / 2.0)	* HypergeometricFunctions._₁F₁((1 - n) / 2.0, 1.5, x^2) +
# 					SpecialFunctions.gamma(-0.5 * n)		* HypergeometricFunctions._₁F₁(-0.5 * n,      0.5, x^2)
# 	) / (2.0 * SpecialFunctions.gamma(-n))

# 	if isnan(res) && !(n isa BigInt || x isa BigFloat)
# 		return T(Hermite2(BigInt(n), BigFloat(x)))
# 	end
# 	return res
# end


# function logpdf_noncental_t(t::T, df, location) where T<:Number
# 	# generated with mathematica
# 	res = log((2^df * df^(1 + df / 2.0) * (df + t^2)^((-1 - df)/2.)*SpecialFunctions.gamma((1 + df)/2.) *
# 		Hermite2(-1 - df,-((location*t)/(sqrt(2)*sqrt(df + t^2)))))/(exp(location^2/2.)*pi))
# 	if isnan(res) && !(t isa BigFloat)
# 		return T(logpdf_noncental_t(BigFloat(t), big(df), big(location)))
# 	end
# 	return res
# end
