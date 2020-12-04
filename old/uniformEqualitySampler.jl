using Turing, Plots#, StatsPlots, MCMCChains
include("src/UniformConditionalPartitionDistribution.jl")
include("src/loglikelihood.jl")

@model function bfvar_gamma(n::Vector{Float64}, b::Vector{Float64}, α::Vector{Float64}, ::Type{T} = Float64) where {T}

	# gammas = Vector{T}(undef, length(n))
	# for i in eachindex(gammas)
	# 	gammas[i] ~ Gamma(α[i], 1)
	# end
	# ρ = gammas ./ sum(gammas)
	ρ ~ Dirichlet(α)
	τ ~ InverseGamma(1, 1)
	Turing.@addlogprob! myloglikelihood(n, b, ρ, τ)
	σ = length(n) * τ .* ρ
	return (σ, )
end

@model function bfvar_gamma_eq(n::Vector{Float64}, b::Vector{Float64}, α::Vector{Float64}, ::Type{T} = Float64) where {T}

	k = length(n)
	# equal_indices = TArray(Int, k)
	# equal_indices[1:end] .= 1
	equal_indices = ones(Int, k)
	for i in eachindex(equal_indices)
		equal_indices[i] ~ UniformConditionalPartitionDistribution(equal_indices, i)
	end

	τ ~ InverseGamma(1, 1)
	ρ ~ Dirichlet(α)
	ρ2 = Vector{T}(undef, k)
	for i in 1:k
		ρ2[i] = mean(ρ[equal_indices .== equal_indices[i]])
	end
	Turing.@addlogprob! myloglikelihood(n, b, ρ2, τ)
	σ = length(n) * τ .* ρ2
	return (σ, ρ2)
	# gammas = Vector{T}(undef, k)
	# for i in 1:k
	# 	gammas[i] ~ Gamma(α[i], 1)
	# end
	# s = sum(gammas)
	# ρ = Vector{T}(undef, k)
	# for i in 1:k
	# 	ρ[i] = mean(gammas[equal_indices .== equal_indices[i]])
	# end
	# ρ = [mean(gammas[findall(equal_indices[i] .== equal_indices)]) for i in eachindex(equal_indices)]
	# ρ ./= sum(ρ)


	# Turing.@addlogprob! myloglikelihood(n, b, ρ, τ)

end

get_eq_ind_nms(samples) = filter(x->startswith(string(x), "equal_indices"), samples.name_map.parameters)

function compute_post_prob_eq(samples)
	eq_ind_nms = get_eq_ind_nms(samples)
	s = size(samples[eq_ind_nms])
	samps = reshape(samples[eq_ind_nms].value.data, :, s[2])
	n_samps, n_groups = size(samps)
	probs = zeros(Float64, n_groups, n_groups)
	for row in eachrow(samps)
		for j in eachindex(row)
			idx = j .+ findall(==(row[j]), row[j+1:end])
			probs[idx, j] .+= 1.0
		end
	end
	return probs ./ n_samps
end

function count_models(samples)
	eq_ind_nms = get_eq_ind_nms(samples)
	s = size(samples[eq_ind_nms])
	samps = reshape(samples[eq_ind_nms].value.data, :, s[2])
	n_samps, n_groups = size(samps)
	dd = Dict{String, Int}()
	for row in eachrow(samps)
		key = join(string.(Int.(row)))
		if haskey(dd, key)
			dd[key] += 1
		else
			dd[key] = 1
		end
	end
	return dd
end

get_eq_nms(m) = filter(startswith("equal_indices["), string.(m.name_map.parameters))
get_rho_nms(m) = filter(startswith("ρ["), string.(m.name_map.parameters))
get_gamma_nms(m) = filter(startswith("gammas["), string.(m.name_map.parameters))
get_tau(m) = m["τ"].data
function get_rho(model)
	nms = get_rho_nms(model)
	nms_eq = get_eq_nms(model)
	isempty(nms_eq) && return model[nms].value.data

	rhos = model[nms].value.data
	eqs  = Int.(model[nms_eq].value.data)
	rhos2 = similar(rhos)
	for j in axes(rhos2, 2)
		for i in axes(rhos2, 1)
			rhos2[i, j] = mean(rhos[i, eqs[i, :] .== eqs[i, j]])
		end
	end
	return rhos2
end

function get_sd(m, s_tau, s_rho)
	k = size(s_rho)[2]
	return 1 ./ sqrt.(k .* s_tau .* s_rho)
end
get_sd(m) = get_sd(m, get_tau(m), get_rho(m))

function compute_means(model)

	s_tau = get_tau(model)
	s_rho = get_rho(model)
	s_sd  = get_sd(model, s_tau, s_rho)

	return Dict{Symbol, Vector{Float64}}(
		:τ => [mean(s_tau)],
		:ρ => vec(mean(s_rho, dims = 1)),
		:σ => vec(mean(s_sd, dims = 1))
	)
end

function trace_plots(model)

	tau = get_tau(model)
	rho = get_rho(model)
	sd  = get_sd(model, tau, rho)

	k = size(rho)[2]

	plots_sd  = [plot(sd[:, i],		title = "sigma $i",	legend = false) for i in axes(sd, 2)]
	plots_rho = [plot(rho[:, i],	title = "rho $i", 	legend = false) for i in axes(rho, 2)]
	plots_tau =  plot(tau, 			title = "tau", 		legend = false);

	l = @layout [
		grid(2, k)
		a
	]
	plot(plots_sd..., plots_rho..., plots_tau, layout = l)
end

function fit_base_model(sds::Vector{Float64}, ns::Vector{Int}, ϵ::Float64 = 0.05, leapfrog::Int = 10; no_samples::Int = 10_000)

	k  = length(sds)
	ss = (sds .* ((ns .- 1) ./ ns)).^2
	n  = (ns .- 1) ./ 2
	b  = ns .* ss
	α  = ones(Float64, length(ss))

	# for some reason, it's really hard to pass these
	obs_tau      = 1 ./ sds .^ 2
	mean_obs_tau = mean(obs_tau)
	obs_rho      = obs_tau ./ (k * mean_obs_tau)
	obs_gamma    = obs_rho
	init_theta   = vcat(mean_obs_tau, obs_gamma)

	model_gamma = bfvar_gamma(n, b, α)
	spl = Gibbs(HMC(ϵ, leapfrog, :τ, :ρ))#HMC(0.2, 3, :τ), HMC(0.2, 3, :gammas))

	# varinfo = Turing.VarInfo(model_gamma);
	# # model_gamma(varinfo, Turing.SampleFromPrior(), Turing.PriorContext((ρ = obs_rho, τ = mean_obs_tau)));
	# model_gamma(varinfo, spl, Turing.PriorContext((ρ = obs_rho, τ = mean_obs_tau)));
	# init_theta = varinfo[Turing.SampleFromPrior()]
	# sample(model_gamma, HMC(0.01, 1), 100, init_theta = init_theta)

	samples_gamma = sample(model_gamma, spl, no_samples)
	# means = Dict(

	# )


	return samples_gamma

end

# @code_warntype model_gamma.f(
#     Random.GLOBAL_RNG,
#     model_gamma,
#     Turing.VarInfo(model_gamma),
#     Turing.SampleFromPrior(),
#     Turing.DefaultContext(),
#     model_gamma.args...,
# )

function fit_model(sds::Vector{Float64}, ns::Vector{Int}, ϵ::Float64 = 0.05, leapfrog::Int = 10; no_samples::Int = 10_000)

	k  = length(sds)
	ss = (sds .* ((ns .- 1) ./ ns)).^2
	n  = (ns .- 1) ./ 2
	b  = ns .* ss
	α  = ones(Float64, length(ss))

	spl = Gibbs(PG(20, :equal_indices), HMC(ϵ, leapfrog, :τ, :ρ))#HMC(0.2, 3, :τ), HMC(0.2, 3, :gammas))
	model_gamma_eq = bfvar_gamma_eq(n, b, α)
	samples_gamma_eq = sample(model_gamma_eq, spl, no_samples)
	eq_probs = compute_post_prob_eq(samples_gamma_eq)
	model_probs = count_models(samples_gamma_eq)

	return eq_probs, model_probs, samples_gamma_eq

end

trueModels = [
	Float64[1, 2, 3],
	Float64[1, 2, 2],
	Float64[1, 1, 1]
]
k = length(trueModels[1])
ns = 100 .* ones(Int, k)

fit123 = fit_base_model(trueModels[1], ns)
compute_means(fit123)
trace_plots(fit123)

fit123_eq = fit_model(trueModels[1], ns)
fit123_eq[1]
fit123_eq[2]
compute_means(fit123_eq[3])
trace_plots(fit123_eq[3])


fits = fit_model.(trueModels, Ref(ns))
fits[1][1]
fits[1][2]
fits[2][1]
fits[2][2]
fits[3][1]
fits[3][2]

fits[1][3]

u = sample_all([1, 1, 1], 1)
mm =

g = rand(Gamma(1, 1), 3)
r = [mean(g[findall(u[i] .== u)]) for i in eachindex(u)]
p = r ./ sum(r)
g ./ sum(g)

# p_eq, p_models, samples = fit_model(Float64[1, 2, 3], 1000)
# p_eq, p_models, samples = fit_model(Float64[1, 2, 2], 1000)


# spl = Gibbs(PG(20, :equal_indices), MH(:τ), MH(:gammas))#HMC(0.2, 3, :τ), HMC(0.2, 3, :gammas))
# spl = Gibbs(MH(:equal_indices), MH(:τ), MH(:gammas))#HMC(0.2, 3, :τ), HMC(0.2, 3, :gammas))
# spl = Gibbs(PG(20, :equal_indices), MH(:τ), MH(:gammas))#HMC(0.2, 3, :τ), HMC(0.2, 3, :gammas))
# spl = Gibbs(PG.(20, Symbol.("equal_indices[" .* string.(1:3) .* "]"))..., MH(:τ), MH(:gammas))
# spl = Gibbs(MH(:equal_indices), HMC(0.2, 3, :τ, :gammas))#HMC(0.2, 3, :τ), HMC(0.2, 3, :gammas))
# model_gamma_eq = bfvar_gamma_eq(n, b, α)
# samples_gamma_eq = sample(model_gamma_eq, spl, 10_000)

# compute_post_prob_eq(samples_gamma_eq)
# count_models(samples_gamma_eq)
