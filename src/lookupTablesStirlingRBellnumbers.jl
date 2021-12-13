#=

	Precompute tables of r-stirlings numbers and bell numbers

	Note that the order here matters, _stirlings2r_table_BigInt must be computed before _bellnumr_table_BigInt


=#



const _r_stirling2r_R_MAX = 15
const _r_stirling2r_N_MAX = 15
const _stirlings2r_table_BigInt = Vector{BigInt}(undef, _r_stirling2r_R_MAX * _r_stirling2r_N_MAX * (_r_stirling2r_N_MAX + 1) ÷ 2)

function _stirling2r_index(n::T, k::T, r::T) where T <: Integer
	# map the triangular array to a linear index
	# k moves fastest then n, then r.
	n = n - r - 1
	k = k - r
	r_offset	= r * _r_stirling2r_N_MAX * (_r_stirling2r_N_MAX + 1) ÷ 2
	nk_offset	= n * (n - 1) ÷ 2 + 1 + k - 1
	return r_offset + nk_offset
end

for r in range(	eltype(_stirlings2r_table_BigInt)(0),		length = _r_stirling2r_R_MAX),
	n in range(	eltype(_stirlings2r_table_BigInt)(r+2),		length = _r_stirling2r_N_MAX),
	k in 		eltype(_stirlings2r_table_BigInt)(r+1):n-1

	_stirlings2r_table_BigInt[_stirling2r_index(n, k, r)] = EqualitySampler._stirlings2r_precompute(n, k, r)
end

# sufficient for all simulation sizes (this could be more clever though)
const _bellnumr_table_BigInt = Matrix{BigInt}(undef, 31, 31);
for n in range(BigInt(5), length = size(_bellnumr_table_BigInt, 1)), r in range(BigInt(0), length = size(_bellnumr_table_BigInt, 2))
	_bellnumr_table_BigInt[r+1, n-4] = bellnumr_inner(n, r);
end
