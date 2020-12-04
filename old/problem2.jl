using Turing, DynamicPPL, FillArrays, Plots

function multivariate_normal_likelihood(obs_mean, obs_var, pop_mean, pop_sds, n)
	# efficient evaluation of log likelihood multivariate normal given sufficient statistics
	result = length(obs_mean) * log(2 * float(pi))
	for i in eachindex(obs_mean)
		pop_prec_i = 1.0 / pop_sds[i] / pop_sds[i]
		result +=
			2 * log(pop_sds[i]) +
			obs_var[i] * pop_prec_i +
			(pop_mean[i] - 2 * obs_mean[i]) * pop_mean[i] * pop_prec_i

	end
	return - n / 2 * result
end

# function multivariate_normal_likelihood0(obs_mean, obs_var, pop_mean, pop_sds, n)
# 	# efficient evaluation of log likelihood multivariate normal given sufficient statistics
# 	pop_prec = 1 ./ (pop_sds .^2)
# 	return - n / 2 * (
# 		length(pop_sds) * log(2 * float(pi)) +
# 		2 * sum(log, pop_sds) +
# 		LinearAlgebra.dot(obs_var,  pop_prec) -
# 		2 * LinearAlgebra.dot(obs_mean, pop_prec .* pop_mean) +
# 		LinearAlgebra.dot(pop_mean, pop_prec .* pop_mean)
# 	)
# end

# obs_mean, obs_var, pop_mean, pop_sds = obsmu, obsmu2, collect(1:3), collect(2:4)
# multivariate_normal_likelihood(obs_mean, obs_var, pop_mean, pop_sds, n)
# multivariate_normal_likelihood0(obs_mean, obs_var, pop_mean, pop_sds, n)

function plottrace(mod, chn)

	gen = generated_quantities(mod, chn);
	τ = chn[:τ].data
	rhos = filter(startswith("ρ"), string.(chn.name_map.parameters))
	ρ2 = similar(chn[rhos].value.data)
	σ = similar(ρ2)
	for i in eachindex(τ)
		σ[i, :] = gen[i][1]
		ρ2[i, :] = gen[i][2]
	end

	plots_sd  = [plot(σ[:, i],	title = "σ $i",	legend = false) for i in axes(σ, 2)]
	plots_rho = [plot(ρ2[:, i],	title = "ρ $i", legend = false) for i in axes(ρ2, 2)]
	plots_tau =  plot(τ, 		title = "τ", 	legend = false);

	l = @layout [
		grid(2, k)
		a
	]
	return plot(plots_sd..., plots_rho..., plots_tau, layout = l)
end

@model function model_both(x, usesufficientstatistics, ::Type{T} = Float64) where {T}

	if usesufficientstatistics
		n, k = x[3], x[4]
	else
		n, k = size(x)
	end

	τ ~ InverseGamma(1, 1)
	ρ ~ Dirichlet(ones(k))
	μ ~ filldist(Normal(0, 5), k)

	# sample equalities among the sds
	equal_indices = ones(Int, k)
	equal_indices[1] ~ Categorical([1])
	equal_indices[2] ~ Categorical([2/5, 3/5])
	equal_indices[3] ~ Categorical(equal_indices[2] == 1 ? [.5, 0, .5] : 1/3 .* ones(3))

	# reassign rho to conform to the sampled equalities
	ρ_c = Vector{T}(undef, k)
	for i in 1:k
		# two sds are equal if equal_indices[i] == equal_indices[j]
		# if they're equal, ρ2[i] is the average of all ρ[j] ∀ j ∈ 1:k such that equal_indices[i] == equal_indices[j]
		ρ_c[i] = mean(ρ[equal_indices .== equal_indices[i]])
	end
	# k * τ .* ρ2 gives the precisions of each group
	σ_c = 1 ./ sqrt.(k * τ .* ρ_c)

	if usesufficientstatistics
		if !isa(_context, Turing.PriorContext)
			Turing.@addlogprob! multivariate_normal_likelihood(x[1], x[2], μ, σ_c, n)
		end
	else
		for i in axes(x, 1)
			x[i, :] ~ MvNormal(μ, σ_c)
		end
	end
	return (σ_c, ρ_c)
end


# simulate data -- all sds different
n = 100
k = 3
sds = collect(1.0 : 2 : 2k) # 1, 3, 5, ...
D = MvNormal(sds)
x = permutedims(rand(D, n))

# fit model -- MvNormal
mod_eq = model_both(x, false)
spl_eq = Gibbs(PG(20, :equal_indices), HMC(0.02, 10, :τ, :ρ, :μ))
chn_eq = sample(mod_eq, spl_eq, 2_000);

plottrace(mod_eq, chn_eq)

# fit model -- fast logposterior
obsmu  = vec(mean(x, dims = 1))
obsmu2 = vec(mean(x->x^2, x, dims = 1))

mod_eq_ss = model_both((obsmu, obsmu2, n, k), true)
spl_eq_ss = Gibbs(PG(20, :equal_indices), HMC(0.02, 10, :τ, :ρ, :μ))
chn_eq_ss = sample(mod_eq_ss, spl_eq_ss, 2_000);

plottrace(mod_eq_ss, chn_eq_ss)

# log posteriors are identical!
function mylogjoint(x, model)
	vi = VarInfo(model)
	vi[SampleFromPrior()] = x
	return logjoint(model, vi)
end
ord =  [10, collect(7:9)..., collect(4:6)..., collect(1:3)...]
nms = chn_eq.name_map.parameters
mylogjoint(chn_eq_ss[nms].value.data[42, ord], mod_eq)
mylogjoint(chn_eq_ss[nms].value.data[42, ord], mod_eq_ss)
