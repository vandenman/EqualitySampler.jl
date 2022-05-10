"""

adapted from https://github.com/richarddmorey/BayesFactor/blob/51e21437a41d04f682899919e66ee4c8b07a2cf9/pkg/BayesFactor/R/meta-ttest-utility.R#L127-L151
"""
function independent_samples_bf(t, N, df, rscale)

	nullLike = Distributions.logpdf(Distributions.TDist(df), t)
	# logPriorProbs = pcauchy(c(upper, lower), scale = rscale, log.p = TRUE)
	# prior.interval = logExpXminusExpY(logPriorProbs[1], logPriorProbs[2])

	delta_est = t / sqrt(N)
	mean_delta = sum((delta_est * N) / sum(N))
	scale_delta = 1 / sqrt(sum(N))

	log_const = meta_t_like(mean_delta, t, N, df, rscale; return_log = true)
	# log_const = meta_t_like_2(mean_delta, t, N, df, rscale; return_log = true)

	# hide messages about the tails of the t-distribution not being computed with full precision.
	# see https://stackoverflow.com/questions/39183938/rs-t-distribution-says-full-precision-may-not-have-been-achieved
	# (integral::Float64, err::Float64), _ = IOCapture.capture() do
	# 	QuadGK.quadgk(delta -> meta_t_like(
	# 	delta, t, N, df, rscale, log_const, mean_delta, scale_delta; return_log = false
	# ), -Inf, Inf, rtol=1e-8)
	# end
	integral::Float64, err::Float64 = Suppressor.@suppress QuadGK.quadgk(
		delta -> meta_t_like(delta, t, N, df, rscale, log_const, mean_delta, scale_delta; return_log = false),
		-Inf, Inf, rtol=1e-8
	)

	# @show t, N, df, rscale, log_const, mean_delta, scale_delta
	# integral, err = QuadGK.quadgk(delta -> meta_t_like_2(
	# 	delta, t, N, df, rscale, log_const, mean_delta, scale_delta; return_log = false
	# ), -Inf, Inf, rtol=1e-8)

	val = log(integral * scale_delta) + log_const #=- prior.interval=# - nullLike
	# err = exp(log(intgl[[2]]) - val)

	return (logbf = val, properror = err, #=method = "quadrature"=#)
end

"""
adapted from https://github.com/richarddmorey/BayesFactor/blob/51e21437a41d04f682899919e66ee4c8b07a2cf9/pkg/BayesFactor/R/meta-ttest-utility.R#L153-L162
"""
function meta_t_like(delta, t, N, df, rscale = 1.0, log_const = 0.0, shift = 0.0, scale = 1.0; return_log::Bool = false)

	logval =
		Distributions.logpdf(Distributions.NoncentralT(df, (scale * delta + shift) * sqrt(N)), t) +
		Distributions.logpdf(Distributions.Cauchy(0.0, rscale), scale * delta + shift) +
		-log_const

	return return_log ? logval : exp(logval)

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
function ttest_test(t::Real, n1::Integer, n2::Integer = 0, rscale::Real = sqrt(2) / 2)

	n1, n2 = promote(n1, n2)
	if iszero(n2)
		nu = n1 - one(n1)
		n  = n1
	else
		nu = n1 + n2 - one(n1) - one(n1)
		n  = exp(log(n1) + log(n2) - log(n1 + n2))
	end

	return independent_samples_bf(t, n, nu, rscale)
end

function ttest_test(x::AbstractVector{T}, y::AbstractVector{T}, rscale::Real = sqrt(2) / 2) where {T <: Real}
	t = get_t_statistic(x, y)
	n1, n2 = length(x), length(y)
	return ttest_test(t, n1, n2, rscale)
end

function get_t_statistic(x::AbstractVector{T}, y::AbstractVector{T}) where {T <: Real}

	nx, ny = length(x), length(y)
	@assert nx == ny
	mx, my = mean(x), mean(y)
	vx, vy = var(x),  var(y)

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

westfall_test(f::StatsModels.FormulaTerm, df::DataFrames.DataFrame; kwargs...) = westfall_test(SimpleDataSet(f, df); kwargs...)
westfall_test(y::AbstractVector{<:AbstractFloat}, g::AbstractVector{<:Integer}; kwargs...) = westfall_test(SimpleDataSet(y, g); kwargs...)
westfall_test(y::AbstractVector{<:AbstractFloat}, g::AbstractVector{<:UnitRange{<:Integer}}; kwargs...) = westfall_test(SimpleDataSet(y, g); kwargs...)

function westfall_test(
	df::SimpleDataSet,
	rscale::Float64 = sqrt(2.0) / 2.0
)

	no_groups = length(df.g)
	logbf_mat     = LinearAlgebra.LowerTriangular(zeros(Float64, no_groups, no_groups))
	post_odds_mat = LinearAlgebra.LowerTriangular(zeros(Float64, no_groups, no_groups))

	# this could be done on a log scale
	pH0 = 1 / 2^(2 / length(no_groups))
	prior_odds = pH0 / (1 - pH0)
	log_prior_odds = log(prior_odds)

	for j in 1:no_groups-1, i in j+1:no_groups
		logbf_mat[i, j]     = ttest_test(view(df.y, df.g[j]), view(df.y, df.g[i]), rscale).logbf
		post_odds_mat[i, j] = logbf_mat[i, j] + log_prior_odds
	end

	return (log_posterior_odds_mat = post_odds_mat, logbf_matrix = logbf_mat, log_prior_odds = log_prior_odds)

end

# R > dput(dt(seq(-5, 5, .1), 3, 1.5, log = TRUE)) # to get ref
# ref = (-8.8480569704851, -8.77187815307963, -8.69431005312623, -8.61530473450975,
# -8.53481197728393, -8.45277915291473, -8.36915109385416, -8.28386995768451,
# -8.19687508626041, -8.10810286048487, -8.01748655166475, -7.92495617074924,
# -7.83043831723916, -7.73385603014917, -7.63512864417141, -7.53417165515144,
# -7.43089660019664, -7.32521095926271, -7.21701808697562, -7.106217185444,
# -6.99270333466955, -6.87636759196045, -6.75709719181388, -6.63477586738158,
# -6.50928433167534, -6.38050096195239, -6.24830274025473, -6.11256651989266,
# -5.97317069480851, -5.82999737383521, -5.68293517706918, -5.53188279180889,
# -5.37675345227604, -5.21748052188182, -5.05402437795931, -4.88638080009038,
# -4.71459105279181, -4.53875381063014, -4.35903898595045, -4.1757033672766,
# -3.98910774860385, -3.79973489403872, -3.60820724736284, -3.4153027761056,
# -3.22196677678652, -3.02931698747339, -2.83863908698046, -2.65136982569098,
# -2.46906579342874, -2.29335729138328, -2.12588884962351, -1.96825030031008,
# -1.82190445872725, -1.68811877461147, -1.56790834499338, -1.46199628874703,
# -1.37079497207299, -1.29440854978088, -1.23265446716051, -1.18509952430918,
# -1.15110511716057, -1.12987630183653, -1.12051011790924, -1.12203980537674,
# -1.13347283521116, -1.15382181852752, -1.18212824443808, -1.21747959280601,
# -1.25902070525565, -1.30596043232637, -1.35757456959419, -1.41320600606794,
# -1.47226287679807, -1.53421536764092, -1.59859168204178, -1.66497355685235,
# -1.73299161098623, -1.80232072719859, -1.87267560194046, -1.94380654876377,
# -2.01549560399379, -2.0875529571012, -2.15981370983334, -2.23213495581866,
# -2.30439316463255, -2.3764818495566, -2.44830949614564, -2.51979772792442,
# -2.5908796859312, -2.6614985998599, -2.73160652999195, -2.80116326075528,
# -2.87013532846981, -2.9384951675475, -3.00622036105513, -3.07329298308631,
# -3.139699021807, -3.20542787333049, -3.27047189774261, -3.33482602964046,
# -3.39848743647671)
# Distributions.logpdf.(Distributions.NoncentralT(3, 1.5), -5:.1:5) ≈ ref


