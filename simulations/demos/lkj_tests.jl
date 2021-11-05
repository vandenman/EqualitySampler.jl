using Turing, MCMCChains, FillArrays, Plots, Distributions, LinearAlgebra, KernelDensity
import DynamicPPL: @submodel

"""
Marginal distributions of lkj samples

adpated from https://mjskay.github.io/ggdist/reference/lkjcorr_marginal.html
"""
function dlkjcorr_marginal(x, K, η, log = false)
	D = Beta(η - 1 + K / 2, η - 1 + K / 2)
	return log ? logpdf(D, (x + 1) / 2) - log(2) : pdf(D, (x + 1) / 2) / 2
end

@model function manual_lkj2(K, eta, ::Type{T} = Float64) where T

	# based on https://github.com/rmcelreath/rethinking/blob/2acf2fd7b01718cf66a8352c52d001886c7d3c4c/R/distributions.r#L212-L248
	alpha = eta + (K - 2) / 2
	r_tmp ~ Beta(alpha, alpha)

	r12 = 2 * r_tmp - 1
	R = Matrix{T}(undef, K, K)
	R[1, 1] = one(T)
	R[1, 2] = r12
	R[2, 2] = sqrt(one(T) - r12^2.0)

	if K > 2

		y = Vector{T}(undef, K-2)
		# the original implementation overwrites a vector with one of a different size, but that won't work in Turing
		# z = Vector{Vector{T}}(undef, K-2)
		# but this should be more efficient

		z ~ MvNormal(ones(K * (K - 1) ÷ 2 - 1))
		z_idx = 0

		for m in 2:K-1

			z_i = view(z, z_idx + 1:z_idx + m)
			z_idx += m

			i = m - 1
			alpha -= 0.5
			y[i] ~ Beta(m / 2, alpha)

			# R[1:m, m+1] .= sqrt(y[i]) .* z[i] ./ LA.norm(z[i])
			R[1:m, m+1] .= sqrt(y[i]) .* z_i ./ norm(z_i)
			# LA.normalize does not work because of https://github.com/JuliaDiff/ForwardDiff.jl/issues/175
			# R[1:m, m+1] .= sqrt(y[i]) .* LA.normalize(z[i])
			R[m+1, m+1] = sqrt(1 - y[i])

		end
	end
	return UpperTriangular(R)
end

function sample_lkj(K, η, no_samples)

	mod_manual_LKJ = manual_lkj2(K, η)
	chn = sample(mod_manual_LKJ, NUTS(), no_samples)
	gen = generated_quantities(mod_manual_LKJ, MCMCChains.get_sections(chn, :parameters))

	# form the sampled correlation matrices from the cholesky factors
	Rs = map(x-> x'x, gen)

	# extract the samples for the individual correlations
	ρs = Matrix{Float64}(undef, no_samples, K * (K - 1) ÷ 2)
	for i in axes(ρs, 1)
		ρs[i, :] .= [Rs[i][m, n] for m in 2:size(Rs[i], 1) for n in 1:m-1]
	end

	return ρs, chn
end

Ks = (3, 4, 5)
ηs = (1.0, 2.0, 5.0)
no_samples = 10_000
results = Vector{NamedTuple}()
for (K, η) in Iterators.product(Ks, ηs)
	ρs, _ = sample_lkj(K, η, no_samples)
	push!(results, (K = K, η = η, ρs = ρs))
end

plts = Vector{Plots.Plot}(undef, length(results))
xcoords = -1.0:0.05:1.0 # for true density
for i in eachindex(plts)

	K, η, ρs = results[i]
	density_estimate = kde(view(ρs, :, 1); npoints = 2^12, boundary = (-1, 1))
	plt = plot(density_estimate.x, density_estimate.density, title = "K = $K, η = $η", legend = isone(i) ? :outerright : false,
	           label = "sampled")
	plot!(plt, xcoords, dlkjcorr_marginal.(xcoords, K, η), label = "theoretical")
	plts[i] = plt

end

plot(plts..., layout = (length(Ks), length(ηs)))

# use the LKJ to fit a normal

n, p = 100, 4

true_means  = collect(1.0:p)
true_R_chol = rand(LKJCholesky(p, 1.0, 'U'))
true_R      = true_R_chol.U'true_R_chol.U
true_sd     = collect(1.0:p)
true_S_chol = true_R_chol.U * Diagonal(true_sd)
true_S      = true_S_chol'true_S_chol

x = rand(MvNormal(true_means, true_S), n)

@model function mvnormal_turing(x, η, ::Type{T} = Float64) where T

	p, n = size(x)
	means ~ MvNormal(ones(Float64, p))
	sds   ~ filldist(InverseGamma(1.0, 1.0), p)

	R_chol = @submodel $(Symbol("manual_lkj")) manual_lkj2(p, η, T)
	Sigma_chol = UpperTriangular(R_chol * Diagonal(sds))

	if det(Sigma_chol) < sqrt(eps(T))
		Turing.@addlogprob! -Inf
		return (R_chol, )
	end
	Sigma = Sigma_chol'Sigma_chol
	

	for i in 1:n
		x[:, i] ~ MvNormal(means, Sigma)
	end

	return (R_chol, )
end

mod_mvnormal = mvnormal_turing(x, 1.0)
samps = sample(mod_mvnormal, NUTS(), 100)

Sigma_chol = [1.6996185109323887 4.861521935132051 -0.6498621746830178 0.21264491910724162; 0.0 7.27536669578297 0.6005138605151561 0.5164578717483551; 0.0 0.0 0.8337881313363649 -0.3674260700803254; 0.0 0.0 0.0 0.35013387861256073]
