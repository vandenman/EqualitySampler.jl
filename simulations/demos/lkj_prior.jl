import Turing, LinearAlgebra, Distributions, FillArrays

function unbounded_to_bounded(y::AbstractVector{T}) where T

	n = length(y)
	k = (1 + isqrt(1 + 8n)) ÷ 2

	# aa = binomial.(2:10, 2)
	# bb = @. (1 + isqrt(1 + 8aa)) ÷ 2
	# bb == 2:10

	z = tanh.(y)
	result = LinearAlgebra.UpperTriangular(zeros(T, (k, k)))
	l = 1
	result[1, 1] = one(T)
	@inbounds for j in 2:k

		result[1, j] = z[l]
		l += 1
		tmp = result[1, j]^2

		for i in 2:j-1
			# result[i, j] = z[l] * sqrt(1 - sum(x->x^2, view(result, 1:i-1, j), init = 0))
			result[i, j] = z[l] * sqrt(one(T) - tmp)
			tmp += result[i, j]^2
			l += 1
		end
		result[j, j] = sqrt(one(T) - tmp)
		# result[j, j] = sqrt(1 - sum(x->x^2, view(result, 1:j-1, j), init = 0))
	end

	return result
end

function logabsdet_lkj_cholesky(y::AbstractVector{T}, R_chol) where T

	n = size(R_chol, 1)
	tmp0 = -2 * sum(x -> log(cosh(x)), y)
	tmp1 = zero(T)
	@inbounds for j in 1:n-1
		for i in j+1:n
			tmp2 = zero(T)
			for jp in 1:j-1
				tmp2 += R_chol[jp, i]^2
			end
			tmp1 += log(one(T) - tmp2)
		end
	end
	return tmp0 + tmp1 / (one(T) + one(T))
end

function logpdf_lkj_cholesky(R_chol, η = 1.0)
	d = size(R_chol, 1)
	tmp = d + 2η - 2
	sum((tmp - i) * log(@inbounds R_chol[i, i]) for i in 2:d) + Distributions.lkj_logc0(d, η)
end

Turing.@model function manual_lkj2(K, η, ::Type{T} = Float64) where T

	# uses the logpdf to sample from the LKJ prior

	y ~ Turing.filldist(Turing.Flat(), binomial(K, 2))
	R_chol = unbounded_to_bounded(y)
	Turing.@addlogprob! logpdf_lkj_cholesky(R_chol, η)
	Turing.@addlogprob! logabsdet_lkj_cholesky(y, R_chol)

	return R_chol
end

@model function manual_lkj(K, eta, ::Type{T} = Float64) where T

	# uses the sampling process to sample from the LKJ prior

	alpha = eta + (K - 2) / 2
	r_tmp ~ Beta(alpha, alpha)

	r12 = 2 * r_tmp - 1
	R = Matrix{T}(undef, K, K)
	R[1, 1] = one(T)
	R[1, 2] = r12
	R[2, 2] = sqrt(one(T) - r12^2.0)

	if K > 2

		y = Vector{T}(undef, K-2)
		# z = Vector{Vector{T}}(undef, K-2)
		z ~ MvNormal(ones(K * (K - 1) ÷ 2 - 1))
		z_idx = 0
		# @show y, z
		for m in 2:K-1

			z_i = view(z, z_idx + 1:z_idx + m)
			z_idx += m

			i = m - 1
			alpha -= 0.5
			y[i] ~ Beta(m / 2, alpha)

			# R[1:m, m+1] .= sqrt(y[i]) .* z[i] ./ LinearAlgebra.norm(z[i])
			R[1:m, m+1] .= sqrt(y[i]) .* z_i ./ LinearAlgebra.norm(z_i)
			# LinearAlgebra.normalize does not work because of https://github.com/JuliaDiff/ForwardDiff.jl/issues/175
			# R[1:m, m+1] .= sqrt(y[i]) .* LinearAlgebra.normalize(z[i])
			R[m+1, m+1] = sqrt(1 - y[i])

		end
	end

	return LinearAlgebra.UpperTriangular(R)

end