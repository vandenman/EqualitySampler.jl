"""
$(TYPEDSIGNATURES)

Computes the ``r``-Bell numbers.
"""
function bellnumr(n::T, r::T) where T <: Integer

	succes, value = bellnumr_base_cases(n, r)
	succes && return value

	return bellnumr_inner(n, r)
end
bellnumr(n::T, r::U) where {T <: Integer, U <: Integer} = bellnumr(promote(n, r)...)

function bellnumr_inner(n::T, r::T) where T <: Integer

	res = zero(T)
	for k in zero(n):n
		res += stirlings2r(n+r, k+r, r)
	end
	return res

end

"""
$(TYPEDSIGNATURES)

Computes the logarithm of the ``r``-Bell numbers.
"""
logbellnumr(n::T, r::U) where {T <: Integer, U <: Integer} = logbellnumr(promote(n, r)...)
# function logbellnumr(n::T, r::T) where T <: Integer

# 	succes, value = bellnumr_base_cases_big(n, r)
# 	if succes
# 		if T === BigInt
# 			return log(value)
# 		else
# 			return Float64(log(value))
# 		end
# 	end

# 	# function is not type stable without this for some reason?
# 	U = T === BigInt ? BigInt : Float64
# 	return convert(U, LogExpFunctions.logsumexp(logstirlings2r(n+r, k+r, r) for k in 0:n))

# end
# logbellnumr(n::BigInt, r::BigInt) = log(bellnumr(n, r))

# function bellnumr_base_cases_big(n::T, r::T) where T <: Integer

# 	# base cases
# 	(b, value) = bellnumr_base_cases_inner(n, r)
# 	b && return (b, BigInt(value))

# 	# if 0 <= r < size(_bellnumr_table_BigInt, 1) && 5 <= n < size(_bellnumr_table_BigInt, 2)
# 	# 	return (true, _bellnumr_table_BigInt[r+1, n-4])
# 	# end

# 	return (false, zero(BigInt))
# end

# function bellnumr_base_cases_inner(n::T, r::T) where T <: Integer

# 	# TODO: consider generating these with mathematica and using @horner.
# 	# should be trivial for the first 10 or so.

# 	# base cases
# 	n == 0		&& 		return (true, 		one(T))
# 	n == 1		&&		return (true, 		r + one(r))
# 	# https://oeis.org/A002522 offset by 1
# 	n == 2 		&& 		return (true, 		abs2(r + one(r)) + one(r))
# 	# https://oeis.org/A005491 offset by 1
# 	n == 3		&&		return (true, 		T((r + 1)^3 + 3(r + 1) + 1))
# 	# https://oeis.org/A005492 simplify equations for a(n)
# 	n == 4		&&		return (true, 		T(15 + r * (20 + r * (12 + r * (4 + r)))))



# 	return (false, zero(T))
# end

# function bellnumr_base_cases(n::T, r::T) where T <: Integer

# 	# base cases
# 	(b, value) = bellnumr_base_cases_inner(n, r)
# 	b && return (b, value)

# 	# if 0 <= r < size(_bellnumr_table_BigInt, 1) && 5 <= n < size(_bellnumr_table_BigInt, 2)
# 	# 	return (true, T(_bellnumr_table_BigInt[r+1, n-4]))
# 	# end

# 	return (false, zero(T))
# end

function bellnumr_base_cases(n::T, r::T) where T <: Integer

    #=

    generated with mathematica:
    ```mathematica
    m = 11;
    BellNumberR[n_,r_]:=Sum[Binomial[n,i]*StirlingS2[i,k]*r^(n-i),{k,0,n},{i,0,n}]
    tb=Table[BellNumberR[n,r],{n,0,m+1},{r,1,m+1}];
    (*tb//MatrixForm*)
    tb3=Table[FindSequenceFunction[tb[[i]],r], {i, 1, m}];
    tb3//MatrixForm;
    HornerForm/@tb3//MatrixForm;
    coeflistToJulia[x_, i_]:=StringJoin["n == ",ToString[i-1], " && return (true, Base.Math.@horner(r, ",StringReplace[ToString[x],{"{"->"","}"->""}],  "))\n"]
    StringJoin[Table[coeflistToJulia[CoefficientList[tb3[[i]], r], i],{i,1,Length[tb3]}]]
    ```

    =#

    n == 0  && return (true, one(r))
    n == 1  && return (true, T(Base.Math.@horner(r, 1, 1)))
    n == 2  && return (true, T(Base.Math.@horner(r, 2, 2, 1)))
    n == 3  && return (true, T(Base.Math.@horner(r, 5, 6, 3, 1)))
    n == 4  && return (true, T(Base.Math.@horner(r, 15, 20, 12, 4, 1)))
    n == 5  && return (true, T(Base.Math.@horner(r, 52, 75, 50, 20, 5, 1)))
    n == 6  && return (true, T(Base.Math.@horner(r, 203, 312, 225, 100, 30, 6, 1)))
    n == 7  && return (true, T(Base.Math.@horner(r, 877, 1421, 1092, 525, 175, 42, 7, 1)))
    n == 8  && return (true, T(Base.Math.@horner(r, 4140, 7016, 5684, 2912, 1050, 280, 56, 8, 1)))
    n == 9  && return (true, T(Base.Math.@horner(r, 21147, 37260, 31572, 17052, 6552, 1890, 420, 72, 9, 1)))
	n == 10 && return (true, T(Base.Math.@horner(r, 115975, 211470, 186300, 105240, 42630, 13104, 3150, 600, 90, 10, 1)))

    return (false, zero(n))
end



"""
$(TYPEDSIGNATURES)

Computes the Bell numbers.
"""
function bellnum(n::T) where {T <: Integer}
	bellnumr(n, zero(T))
end



function all_logbellnum(n::Integer)

	# would be nice to have this allocate less.
    list = Vector{float(typeof(n))}(undef, n+1)
    logbellnumbers = similar(list)

    list[1] = 0
    logbellnumbers[1] = 0
    for i = 2:n+1
        for j = 1:i - 2
            list[i - j - 1] = LogExpFunctions.logaddexp(list[i - j - 1], list[i - j])
        end
        list[i] = LogExpFunctions.logaddexp(list[1], list[i - 1])
        logbellnumbers[i] = list[i]
    end
    return logbellnumbers
end

function logbellnumr(n::T, r::T) where T<:Integer

	succes, value = bellnumr_base_cases(n, r)
	succes && return log(value)

	# can be computed without allocations?
    iszero(r) && return last(all_logbellnum(n - 1))

    # could add these two lines in all_logbellnum already
    logbellnumbers = reverse!(all_logbellnum(n - 1))
    push!(logbellnumbers, 0)

    LogExpFunctions.logsumexp(
        k * log(r) + SpecialFunctions.logabsbinomial(n, k)[1] + logbellnumbers[k+1]
        for k in zero(n):n
    )

end
