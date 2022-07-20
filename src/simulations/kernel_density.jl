"""
$(TYPEDSIGNATURES)

Compute density estimates for matrix `mat` where columns contains samples of different parameters.
"""
function compute_density_estimate(mat::AbstractMatrix{<:Real}, npoints = 2^12)
	ngroups = size(mat, 2)
	x = Matrix{Float64}(undef, npoints, ngroups)
	y = Matrix{Float64}(undef, npoints, ngroups)

	for (i, col) in enumerate(eachcol(mat))
		k = KernelDensity.kde(col; npoints = npoints)#, boundary = (0, Inf))
		x[:, i] .= k.x
		y[:, i] .= k.density
	end

	return (x = x, y = y)
end

compute_density_estimate(mat::MCMCChains.Chains, args...) = compute_density_estimate(Array(mat), args...)
