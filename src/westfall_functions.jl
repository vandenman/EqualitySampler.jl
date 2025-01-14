"""

adapted from https://github.com/richarddmorey/BayesFactor/blob/51e21437a41d04f682899919e66ee4c8b07a2cf9/pkg/BayesFactor/R/meta-ttest-utility.R#L127-L151
"""
function independent_samples_bf(t, N, df, rscale, pure_julia::Bool = true)

	# @show t, N, df, rscale
	nullLike = Distributions.logpdf(Distributions.TDist(df), t)
	# logPriorProbs = pcauchy(c(upper, lower), scale = rscale, log.p = TRUE)
	# prior.interval = logExpXminusExpY(logPriorProbs[1], logPriorProbs[2])

	delta_est = t / sqrt(N)
	mean_delta = sum((delta_est * N) / sum(N))
	scale_delta = 1 / sqrt(sum(N))

	meta_t_fun = pure_julia ? meta_t_like_pure_julia : meta_t_like
	log_const = meta_t_fun(mean_delta, t, N, df, rscale; return_log = true)

	if isinf(log_const)
		@show mean_delta, t, N, df, rscale
		throw(DomainError("bad log_const for mean_delta=$mean_delta, t=$t, N=$N, df=$df, rscale=$rscale"))
	end

	# log_const = meta_t_like_2(mean_delta, t, N, df, rscale; return_log = true)

	# hide messages about the tails of the t-distribution not being computed with full precision.
	# see https://stackoverflow.com/questions/39183938/rs-t-distribution-says-full-precision-may-not-have-been-achieved
	# (integral::Float64, err::Float64), _ = IOCapture.capture() do
	# 	QuadGK.quadgk(delta -> meta_t_like(
	# 	delta, t, N, df, rscale, log_const, mean_delta, scale_delta; return_log = false
	# ), -Inf, Inf, rtol=1e-8)
	# end

	integral::Float64, err::Float64 = QuadGK.quadgk(
		delta -> meta_t_fun(delta, t, N, df, rscale, log_const, mean_delta, scale_delta; return_log = false),
		-Inf, Inf, rtol=1e-8
	)

	val = log(integral * scale_delta) + log_const #=- prior.interval=# - nullLike
	# err = exp(log(intgl[[2]]) - val)

	return (logbf = val, properror = err, #=method = "quadrature"=#)
end

"""
adapted from https://github.com/richarddmorey/BayesFactor/blob/51e21437a41d04f682899919e66ee4c8b07a2cf9/pkg/BayesFactor/R/meta-ttest-utility.R#L153-L162
"""
function meta_t_like(delta, t, N, df, rscale = 1.0, log_const = 0.0, shift = 0.0, scale = 1.0; return_log::Bool = false)

	logval =
	    (Suppressor.@suppress Distributions.logpdf(Distributions.NoncentralT(df, (scale * delta + shift) * sqrt(N)), t)) +
		Distributions.logpdf(Distributions.Cauchy(0.0, rscale), scale * delta + shift) +
		-log_const

	return return_log ? logval : exp(logval)

end

function meta_t_like_pure_julia(delta, t, N, df, rscale = 1.0, log_const = 0.0, shift = 0.0, scale = 1.0; return_log::Bool = false)

	logval =
		noncentralt_logpdf(t, df, (scale * delta + shift) * sqrt(N)) +
		Distributions.logpdf(Distributions.Cauchy(zero(rscale), rscale), scale * delta + shift) +
		-log_const

	# return return_log ? logval : exp(logval)
	retval = return_log ? logval : exp(logval)
	if isnan(retval) || (isinf(retval) && retval > 0)
		@show retval, delta, t, N, df, rscale, log_const, shift, scale
		throw(DomainError("bad revtal for delta=$delta, t=$t, N=$N, df=$df, rscale=$rscale, log_const=$log_const, shift=$shift, scale=$scale"))
	end
	return retval

end


# Not accurate enough
# """
# identical to meta_t_like up to floating point accuracy, but does not use Rmath.
# """
# function meta_t_like_2(delta, t, N, df, rscale = 1.0, log_const = 0.0, shift = 0.0, scale = 1.0; return_log::Bool = false)

# 	logval =
# 		# Distributions.logpdf(Distributions.NoncentralT(df, (scale * delta + shift) * sqrt(N)), t) +
# 		logpdf_noncental_t(t, df, (scale * delta + shift) * sqrt(N)) +
# 		Distributions.logpdf(Distributions.Cauchy(0.0, rscale), scale * delta + shift) +
# 		-log_const

# 	if isnan(logval)
# 		@show delta, t, N, df, rscale, log_const, shift, scale
# 	end

# 	return return_log ? logval : exp(logval)

# end

# see if this also works instead of meta_t_like, so we can skip IOCapture.capture() which crashes in parallel.
# for HermiteH try out SpecialPolynomials.basis(SpecialPolynomials.Hermite, degree)(value)
# Tried it, but it's not accurate enough
# function foo(delta, t, N, df, rscale = 1.0, log_const = 0.0, shift = 0.0, scale = 1.0; log::Bool = false)
 	# log(1 / (pi * rscale * (1 + (0.0 + delta * scale + shift)^2 / rscale^2))) +
	# 	log((2^df * df^(1 + df / 2.0) * (df + t^2)^((-1 - df)/2.0) * Gamma((1 + df)/2.0) *
	# 	HermiteH(-1 - df,-((Sqrt(n)*(delta*scale + shift)*t)/ - (Sqrt(2)*Sqrt(df + t^2))))) / (exp((n*(delta*scale + shift)^2) / 2.0) * pi))
# end

"""
adapted from https://github.com/richarddmorey/BayesFactor/blob/452624486c111436910061770bd1fa7ea6a69d62/pkg/BayesFactor/R/ttest_tstat.R
"""
function ttest_test(t::Real, n1::Integer, n2::Integer = 0, rscale::Real = sqrt(2) / 2, pure_julia::Bool = true)

	n1, n2 = promote(n1, n2)
	if iszero(n2)
		nu = n1 - one(n1)
		n  = n1
	else
		nu = n1 + n2 - one(n1) - one(n1)
		n  = exp(log(n1) + log(n2) - log(n1 + n2))
	end

	return independent_samples_bf(t, n, nu, rscale, pure_julia)
end

function ttest_test(x::AbstractVector{T}, y::AbstractVector{T}, rscale::Real = sqrt(2) / 2, pure_julia::Bool = true) where {T <: Real}
	t = get_t_statistic(x, y)
	n1, n2 = length(x), length(y)
	return ttest_test(t, n1, n2, rscale, pure_julia)
end

function get_t_statistic(x::AbstractVector{T}, y::AbstractVector{T}) where {T <: Real}

	nx, ny = length(x), length(y)
	@assert nx == ny
	mx, my = StatsBase.mean(x), StatsBase.mean(y)
	vx, vy = StatsBase.var(x),  StatsBase.var(y)

	# assumes equal variance
	df = nx + ny - 2
	v = 0.0
	if nx > 1
		v += (nx - 1) * vx
	end
	if ny > 1
		v += (ny - 1) * vy
	end
	v /= df
	standard_error = sqrt(v * (1/nx + 1/ny))
	t_statistic = (mx - my) / standard_error
	return t_statistic
end

# TODO: replace SimpleDataSet with the suffstats object?
westfall_test(f::StatsModels.FormulaTerm, df::DataFrames.DataFrame; kwargs...) = westfall_test(SimpleDataSet(f, df); kwargs...)
westfall_test(y::AbstractVector{<:AbstractFloat}, g::AbstractVector{<:Integer}; kwargs...) = westfall_test(SimpleDataSet(y, g); kwargs...)
westfall_test(y::AbstractVector{<:AbstractFloat}, g::AbstractVector{<:UnitRange{<:Integer}}; kwargs...) = westfall_test(SimpleDataSet(y, g); kwargs...)

function westfall_test(
	df::SimpleDataSet,
	rscale::Float64 = sqrt(2.0) / 2.0;
	pure_julia::Bool = true
)

	no_groups = length(df.g)
	logbf_mat     = LinearAlgebra.LowerTriangular(zeros(Float64, no_groups, no_groups))
	post_odds_mat = LinearAlgebra.LowerTriangular(zeros(Float64, no_groups, no_groups))

	# this could be done on a log scale
	pH0 = 1 / 2^(2 / length(no_groups))
	prior_odds = pH0 / (1 - pH0)
	log_prior_odds = log(prior_odds)

	for j in 1:no_groups-1, i in j+1:no_groups
		logbf_mat[i, j]     = ttest_test(view(df.y, df.g[j]), view(df.y, df.g[i]), rscale, pure_julia).logbf
		post_odds_mat[i, j] = logbf_mat[i, j] + log_prior_odds
	end

	return (log_posterior_odds_mat = post_odds_mat, logbf_matrix = logbf_mat, log_prior_odds = log_prior_odds)

end

