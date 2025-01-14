function loglikelihood_suffstats(d::Distributions.Normal, ss::Distributions.NormalStats)
	μ, σ = Distributions.params(d)
	x̄, s, n = ss.m, ss.s2, ss.tw
	return -n / 2.0 * (log(2pi) + 2log(σ)) - 1 / 2.0σ^2 * (s + n * abs2(x̄ .- μ))
end

function loglikelihood_suffstats(d::Distributions.AbstractMvNormal, ss::Distributions.MvNormalStats)
	μ, Σ = Distributions.params(d)
	x̄, S, n = ss.m, ss.s2, ss.tw
	# should check for length mismatches
	p = length(x̄)
	return (
		-n / 2 * (
			p * log(2pi) +
			LinearAlgebra.logdet(Σ) +
			PDMats.invquad(Σ, x̄ .- μ) +
			# PDMats.X_invA_Xt(Σ, (x̄ .- μ)') +
			# (x̄ .- μ)' / Matrix(Σ) * (x̄ .- μ) +
			# LinearAlgebra.tr(Σ \ (S ./ ss.tw))
			LinearAlgebra.tr(Σ \ S) / ss.tw
		)
	)
end
function logpdf_mv_normal_chol_suffstat(x̄, S_chol::LinearAlgebra.UpperTriangular, n, μ, Σ_chol::LinearAlgebra.UpperTriangular)
	d = length(x̄)
	return (
		-n / 2 * (
			d * log(2pi) +
			2 * sum(i->log(@inbounds Σ_chol[i, i]), axes(Σ_chol, 1)) +
			# 2LinearAlgebra.logdet(Σ_chol) +
			sum(x->x^2, (x̄ .- μ)' / Σ_chol) +
			sum(x->x^2, S_chol / Σ_chol)
		)
	)
end
"""
get_normal_dense_chol_suff_stats(x::AbstractMatrix{<:Real})

Returns the sufficients statistics for a multivariate normal with dense covariance matrix.
Note that the cholesky of the sample covariance is the "biased" version (divided by n, instead of n - 1).
"""
function get_normal_dense_chol_suff_stats(x::AbstractMatrix{<:Real})
	ss = Distributions.suffstats(Distributions.MvNormal, x)
	obs_mean, obs_cov_unnormalized, n = ss.m, ss.s2, Int(ss.tw)
	obs_cov_chol = LinearAlgebra.cholesky(obs_cov_unnormalized ./ n).U
	return obs_mean, obs_cov_chol, n
end
