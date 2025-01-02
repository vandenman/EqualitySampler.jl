# TODO: consider making a PR to StatsFuns.jl?

"""
Logarithm of the Hermite function Hₙ(z) for possibly negative n.

Implements the definition from
https://functions.wolfram.com/HypergeometricFunctions/HermiteHGeneral/02/
"""
function logHermiteH(n::Number, z::Number)

	# TODO: for integer n, this can be done more efficiently and accurately.
	# both loggamma terms have analytic expressions.
	# the hypergeometric functions can be computed with recurrence relations.

    z² = abs2(z)
    result = n * log(2) + 1/2 * log(π)

    lg1 = SpecialFunctions.loggamma((1 - n) / 2)
    lg2 = SpecialFunctions.loggamma(   - n  / 2)
    a1 = -n / 2
    a2 = (1 - n) / 2
    b1 = 1 / 2
    b2 = 3 / 2
    c = z²
    add = zero(z)

    if z > 10
        a1 = b1 - a1
        a2 = b2 - a1
        add = z²
        c = -z²
    end

    expterm1 = HypergeometricFunctions._₁F₁(a1, b1, c)
    expterm2 = HypergeometricFunctions._₁F₁(a2, b2, c)
    sign1 = sign(expterm1)
    sign2 = sign(expterm2)

    temp1 = log(abs(expterm1)) - lg1
    temp2 = log(abs(expterm2)) - lg2

    if isinf(temp1) || isinf(temp2)
        a1 = b1 - a1
        a2 = b2 - a1
        add = z²
        c = -z²

        expterm1 = HypergeometricFunctions._₁F₁(a1, b1, c)
        expterm2 = HypergeometricFunctions._₁F₁(a2, b2, c)
        sign1 = sign(expterm1)
        sign2 = sign(expterm2)

        temp1 = log(abs(expterm1)) - lg1
        temp2 = log(abs(expterm2)) - lg2
    end

    positive1 = sign1 > 0
	# ! because it's a subtraction originally, so if it's negative the -s cancel out.
    positive2 = !((sign2 > 0 && z > 0) || (sign2 < 0 && z < 0))
    if positive1 && positive2
        temp3 = LogExpFunctions.logaddexp(temp1, log(2abs(z))        + temp2)
    elseif positive1 && !positive2
        temp3 = LogExpFunctions.logsubexp(temp1, log(2abs(z))   + temp2)
    elseif positive2 && !positive1
        temp3 = LogExpFunctions.logsubexp(log(2abs(z)) + temp2,   temp1)
    else
        @show n, z
        throw(ArgumentError("Both terms are negative"))
    end

    result += add + temp3

    # terrible fallback
    if !(result isa BigFloat) && isinf(result) && result > zero(result)
        return oftype(z, logHermiteH(big(n), big(z)))
    end

    return result
end

HermiteH(v::Number, z::Number) = exp(logHermiteH(v, z))

function HermiteH(v::Integer, z::Number)

    v > zero(v) && throw(DomainError("Not implemented for v > 0."))
    prev2 = one(z)
    iszero(v) && return prev2

    prev1 = exp(abs2(z)) / 2 * √π * SpecialFunctions.erfc(z)
    isone(-v) && return prev1

    for v′ in oftype(v, -1):-one(v):v#-one(v)

        cur = prev1 * z / v′ - prev2 / 2v′
        prev2, prev1 = prev1, cur
    end

    return prev2

end

function logHermiteH(v::Integer, z::Number)

    v > zero(v) && throw(DomainError("Not implemented for v > 0."))
    prev2 = zero(z)
    iszero(v) && return prev2

    prev1 = abs2(z) - log(2) + log(π)/2 + SpecialFunctions.logerfc(z)
    isone(-v) && return prev1

    # v_next = -(one(v) + one(v))
    # cur = zero(z)
    # for _ in 0:-v-1
    for v′ in oftype(v, -1):-one(v):v#-one(v)

        # v' always negative, prev1 and prev2 always positive
        # cur = prev1 * x / v′ - prev2 / 2v′
        # cur = prev1 * x / v′ + prev2 / 2|v′|
        if z < zero(z)
            cur = LogExpFunctions.logaddexp(
                prev1 + log(z / v′),
                prev2 - log(abs(2v′))
            )
        else
            # TODO: merge - signs where possible, e.g., v′ is always negative so no need for logsubexp.
            # also, prev1 and prev2 are strictly positive
            abs_v′ = abs(v′)
            cur = LogExpFunctions.logsubexp(
                prev2 - log(2abs_v′),
                prev1 + log(z / abs_v′)
            )
            # cur = LogExpFunctions.logsubexp(LogExpFunctions.logsubexp(prev1 + log(x), v′), prev2 - log(2v′))
        end
        # @show cur, v_next
        # v_next -= one(v)
        prev2, prev1 = prev1, cur
    end

    return prev2

end

# function logHermiteH_old(n::Number, z::Number)
#     z² = abs2(z)
#     result = n * log(2) + 1/2 * log(π)

#     lg1 = SpecialFunctions.loggamma((1 - n) / 2)
#     lg2 = SpecialFunctions.loggamma(   - n  / 2)
#     a1 = -n / 2
#     a2 = (1 - n) / 2
#     b1 = 1 / 2
#     b2 = 3 / 2
#     c = z²
#     add = zero(z)

#     if z > 10 # TODO: need a better value here.
#         a1 = b1 - a1
#         a2 = b2 - a1
#         add = z²
#         c = -z²
#     end

#     expterm1 = HypergeometricFunctions._₁F₁(a1, b1, c)
#     expterm2 = HypergeometricFunctions._₁F₁(a2, b2, c)
#     if expterm1 < 0 || expterm2 < 0
#         @show n, z
#     end

#     temp1 = log(HypergeometricFunctions._₁F₁(a1, b1, c)) - lg1
#     temp2 = log(HypergeometricFunctions._₁F₁(a2, b2, c)) - lg2

#     if isinf(temp1) || isinf(temp2)
#         a1 = b1 - a1
#         a2 = b2 - a1
#         add = z²
#         c = -z²
#         temp1 = log(HypergeometricFunctions._₁F₁(a1, b1, c)) - lg1
#         temp2 = log(HypergeometricFunctions._₁F₁(a2, b2, c)) - lg2
#     end

#     if z < zero(z)
#         temp3 = LogExpFunctions.logaddexp(temp1, log(2*abs(z)) + temp2)
#     else
#         # TODO: unclear if this can throw an error due to numerical issues
#         temp3 = LogExpFunctions.logsubexp(temp1, log(2z)       + temp2)
#     end
# 	@show z, result, temp1, temp2, temp3
#     result += add + temp3
#     return result
# end

# function logHermiteH(n::Number, z::Number)

#     z² = abs2(z)
#     result = n * log(2) + 1/2 * log(π)
#     if z <= 15
#         result += log(abs(
#                  HypergeometricFunctions._₁F₁(   - n  / 2, 1 / 2, z²) / SpecialFunctions.gamma((1 - n) / 2)  -
#             2z * HypergeometricFunctions._₁F₁((1 - n) / 2, 3 / 2, z²) / SpecialFunctions.gamma(   - n  / 2)
#         ))
#     else
#         result += abs2(z) + log(abs(
#                  HypergeometricFunctions._₁F₁(1 / 2 +      n  / 2, 1 / 2, -z²) / SpecialFunctions.gamma((1 - n) / 2)  -
#             2z * HypergeometricFunctions._₁F₁(3 / 2 - (1 - n) / 2, 3 / 2, -z²) / SpecialFunctions.gamma(   - n  / 2)
#         ))
#     end

# 	isinf(result) && result > zero(result) && return -result

#     return result

# end

"""
Native julia implementation of the noncentral t-distribution logpdf.

See also [`noncentralt_pdf`](@ref)
"""
function noncentralt_logpdf(x::Number, ν::Number, δ::Number)

    x² = abs2(x)
    ν * log(2)+
        -abs2(δ)/2 +
        (1 + ν / 2) * log(ν) +
        (-(1 + ν) / 2) * log(x² + ν) +
        SpecialFunctions.loggamma((1 + ν) / 2) +
        logHermiteH(-1 - ν, -x*δ / (sqrt(2) * sqrt(x² + ν))) - log(pi)

end
"""
Native julia implementation of the noncentral t-distribution pdf.
Note that this just calls exp(noncentraltdist_logpdf(x, δ, ν)).

See also [`noncentralt_logpdf`](@ref)
"""
noncentralt_pdf(x::Number, δ::Number, ν::Number) = exp(noncentralt_logpdf(x, δ, ν))

