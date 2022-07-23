#=

	Precompute tables of r-stirlings numbers and bell numbers

	Note that the order here matters, _stirlings2r_table_BigInt must be computed before _bellnumr_table_BigInt

	TODO:
		- create these using the recurrence relations and see if that is faster (it should be).
		- time the impact on package load time

		- make the table sizes mutable by users
		- create an API to fill them
		- do not fill them on package load
		- use/ find/ create something efficient for triangular matrices and vectors thereof
=#


# TODO: it'd be great if we could do without these globals
# can we just check the size in the functions that index in _stirlings2r_table_BigInt?
const _r_stirling2r_R_MAX = 15
const _r_stirling2r_N_MAX = 15

function _stirling2r_index(n::T, k::T, r::T, _r_stirling2r_R_MAX = 15, _r_stirling2r_N_MAX = 15) where T <: Integer
	# map the triangular array to a linear index
	# k moves fastest then n, then r.
	n = n - r - 1
	k = k - r
	r_offset	= r * _r_stirling2r_N_MAX * (_r_stirling2r_N_MAX + 1) ÷ 2
	nk_offset	= n * (n - 1) ÷ 2 + 1 + k - 1
	return r_offset + nk_offset
end

function _precompute_r_stirlings2_new(_r_stirling2r_R_MAX::Int = 15, _r_stirling2r_N_MAX::Int = 15, T::Type{<:Integer} = BigInt)
	stirlings2r_table = Vector{T}(undef, _r_stirling2r_R_MAX * _r_stirling2r_N_MAX * (_r_stirling2r_N_MAX + 1) ÷ 2)
	@inbounds for r in range(	T(0),		length = _r_stirling2r_R_MAX),
		n in range(	T(r+2),		length = _r_stirling2r_N_MAX),
		k in 		T(r+1):n-1

		if n == r + 2 || k == r + 1
			stirlings2r_table[_stirling2r_index(n, k, r)] = _stirlings2r_precompute(n, k, r)
		else
			prev_1 = k == n - 1 ? one(T) : stirlings2r_table[_stirling2r_index(n - 1, k, r)]
			prev_2 = stirlings2r_table[_stirling2r_index(n - 1, k - 1, r)]
			stirlings2r_table[_stirling2r_index(n, k, r)] = k * prev_1 + prev_2
		end
	end

	return stirlings2r_table
end
const _stirlings2r_table_BigInt = _precompute_r_stirlings2_new()


# sufficient for all simulation sizes (this could be more clever though)
# const _bellnumr_table_BigInt = Matrix{BigInt}(undef, 31, 31);
# for n in range(BigInt(5), length = size(_bellnumr_table_BigInt, 1)), r in range(BigInt(0), length = size(_bellnumr_table_BigInt, 2))
# 	_bellnumr_table_BigInt[r+1, n-4] = bellnumr_inner(n, r);
# end

function _precompute_r_bell_numbers(n_max::Int = 31, r_max::Int = 31, T::Type{<:Integer} = BigInt)
	bellnumr_table = Matrix{T}(undef, n_max, r_max);
	for n in range(T(5), length = size(bellnumr_table, 1)),
		r in range(T(0), length = size(bellnumr_table, 2))
		bellnumr_table[r+1, n-4] = bellnumr_inner(n, r);
	end
	return bellnumr_table
end

function _precompute_r_bell_numbers_new(n_max::Int = 31, r_max::Int = 31, T::Type{<:Integer} = BigInt)

	@inbounds begin

		bellnumr_table = Matrix{T}(undef, n_max, r_max);
		# first compute the bell numbers
		n_start = T(5)
		n_end   = n_start + n_max# - 1

		n_first = T(5)

		# TODO: there is some gain by computing this differently!
		# cache_bellnum = bellnumr_inner.(zero(T):n_end, zero(T))
		# adapted from Combinatorics.bellnum to return all the bell numbers
		cache_bellnum = Vector{T}(undef, n_end + 1)
		cache_bellnum[1] = one(T)
		cache_bellnum[2] = one(T)
		list = Vector{T}(undef, n_end + 1)
		list[1] = 1
		for i = 2:n_end
			for j = 1:i - 2
				list[i - j - 1] += list[i - j]
			end
			list[i] = list[1] + list[i - 1]
			cache_bellnum[i+1] = list[i]
		end
		# @assert cache_bellnum == bellnumr_inner.(zero(T):n_end, zero(T))

		# cache_bellnum = Vector{T}(undef, n_end + 1)
		# cache_bellnum[1] = 1
		# temp = 1
		# for n in 1:n_end
		# 	cache_bellnum[n+1] = binomial(ncache_bellnum[n]
		# end
		cache_bellnum[2] = 1
		bellnumr_table[:, 1] .= cache_bellnum[n_first+1:n_end]#bellnumr_inner.(range(n_first, length = size(bellnumr_table, 1)), zero(T))

		cache_binomials = Vector{T}(undef, n_end + 1) # +1 since k = 0 also counts
		cache_binomials[1] = one(T)
		for k in 1:n_start + 1
			cache_binomials[k+1] = cache_binomials[k] * (n_start + 1 - k + 1) ÷ k
		end
		# @assert cache_binomials[1:n_start+2] == binomial.(n_start + 1, 0:n_start + 1)

		multiplication_cache = Vector{T}(undef, n_end + 1)
		for n in range(n_start + 1, length = size(bellnumr_table, 1)-1)
			for k in 0:n
				multiplication_cache[k+1] = cache_binomials[k+1] * cache_bellnum[n - k + 1]
			end
			for r in range(zero(T), length = size(bellnumr_table, 2))

				bellnumr_table[r+1, n-4] = Base.sum(r^k * multiplication_cache[k+1] for k in 0:n)
				# @assert  bellnumr_table[r+1, n-4] == bellnumr_inner(n, r)

			end

			if n != n_end
				temp1, temp2 = cache_binomials[1], cache_binomials[2]
				for k in 1:n
					temp2 = cache_binomials[k+1]
					cache_binomials[k+1] = temp1 + temp2
					temp1 = temp2
				end
				cache_binomials[n+2] = one(T)
				# @assert cache_binomials[1:n+2] == binomial.(n+1, 0:n+1)
			end
		end
	end

	return bellnumr_table
end

# TODO: this guy drives the load times!
const _bellnumr_table_BigInt = _precompute_r_bell_numbers_new()

# since k = 0, 1, 2, n-3, n-2, n-1, n are trivial to compute, the triangle starts at S1(7, 3)

function _stirling1_index(n::T, k::T) where T<:Integer
	#=
	triangle starts at S1(7, 3) so
	(7,3) -> 1,
	(8,3) -> 2, (8,4) -> 2,
	(9,3) -> 4, (9,4) -> 5, (9,5) -> 6
	etc.
	=#
	return (n - 7) * (n - 6) ÷ 2 + (k - 2)
end

const _stirling1_N_MAX = 50
function _precompute_stirlings1_new(n_max::Int = 50, T::Type{<:Integer} = BigInt)
	stirlings1_table = Vector{T}(undef, _stirling1_index(n_max, n_max - 4))
	for n in T(7):n_max, k in 3:n-4
		# @show n, k
		if n == T(7) || k == 3 || k > n - 5
			stirlings1_table[_stirling1_index(n, k)] = _unsignedstirlings1_precompute(n, k)
		else
			prev_1 = stirlings1_table[_stirling1_index(n - 1, k)]
			prev_2 = stirlings1_table[_stirling1_index(n - 1, k - 1)]
			stirlings1_table[_stirling1_index(n, k)] = (n - 1) * prev_1 + prev_2
		end
		# @assert stirlings1_table[_stirling1_index(n, k)] == _unsignedstirlings1_precompute(n, k)
	end
	return stirlings1_table
end

const _stirlings1_table_BigInt = _precompute_stirlings1_new(_stirling1_N_MAX)

# m1 = [ k < n ? unsignedstirlings1(n, k) : 0 for n in big(0):50, k in big(0):50]
# m2 = [ k < n ? EqualitySampler._unsignedstirlings1_precompute(n, k) : 0 for n in big(0):50, k in big(0):50]
# m1 == m2