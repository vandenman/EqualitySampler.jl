"""
Compute the generalized stirling numbers using equation 1.22 of Pitman (2006).

Pitman, J. (2006). Combinatorial stochastic processes: Ecole d'eté de probabilités de saint-flour xxxii-2002. Springer.
"""
function generalized_stirling_number(α::Real, β::Real, n::Integer, k::Integer)

    T = promote_type(typeof(α), typeof(β))
    (1 <= k <= n && n > 0) || return zero(T)
    k == n && return one(T)

    result = zero(T)
    for j in k:n
        result += stirlings1(n, j) * stirlings2(j, k) * α^(n - j) * β^(j - k)
    end
    return result
end

# this is possible but tedious. it can only return (sign, value)
# function log_generalized_stirling_number(α::Real, β::Real, n::Integer, k::Integer)

#     T = float(promote_type(typeof(α), typeof(β)))
#     (1 <= k <= n && n > 0) || return T(-Inf)
#     k == n && return zero(T)


#     result = zero(T)
#     for j in k:n
#         result += stirlings1(n, j) * stirlings2(j, k) * α^(n - j) * β^(j - k)
#     end
#     return result
# end