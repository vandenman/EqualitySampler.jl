using Turing, Plots

function myloglikelihood(n, b, ρ, τ)

	prec = ρ .* (τ * length(n))
	out =
		-logpdf(Gamma(0.001, 0.001), τ) +
		-log(τ) +
		sum(n .* log.(prec)) +
		-0.5 * sum(prec .* b)
	return out
end

@model function mymodel(n::Vector{Float64}, b::Vector{Float64}, α::Vector{Float64}, ::Type{T} = Float64) where {T}

	τ ~ Gamma(0.001, 0.001)
	gammas = Vector{T}(undef, length(n))
	for i in eachindex(gammas)
		gammas[i] ~ Gamma(α[i], 1)
	end
	ρ = gammas ./ sum(gammas)
	Turing.@addlogprob! myloglikelihood(n, b, ρ, τ)

end


@model function mymodel2(n::Vector{Float64}, b::Vector{Float64}, α::Vector{Float64}, ::Type{T} = Float64) where {T}

    τ ~ Gamma(0.001, 0.001)
    ρ ~ Dirichlet(α)
	Turing.@addlogprob! myloglikelihood(n, b, ρ, τ)

end


sds = Float64[1, 2, 3]
ns  = 1_000 .* ones(Int, 3)

# prepare data
k  = length(sds)
ss = (sds .* ((ns .- 1) ./ ns)).^2
n  = (ns .- 1) ./ 2
b  = ns .* ss
α  = ones(Float64, length(ss))

samples = sample(mymodel(n, b, α), HMC(0.01, 10), 10_000)
nms_gamma = filter(startswith("gammas["), string.(samples.name_map.parameters))
samples_gamma = samples[nms_gamma].value.data#, 10_000, 3)
samples_tau   = samples[:τ].data
samples_sd    = similar(samples_gamma)
for i in eachindex(samples_tau)
    samples_sd[i, :] .= 1 ./ sqrt.(k * samples_tau[i] .* (samples_gamma[i, :]) ./ sum(samples_gamma[i, :]))
end
mean(samples_sd, dims = 1)


sds = Float64[1, 2, 3]
ns  = 1_000 .* ones(Int, 3)

# prepare data
k  = length(sds)
ss = (sds .* ((ns .- 1) ./ ns)).^2
n  = (ns .- 1) ./ 2
b  = ns .* ss
α  = ones(Float64, length(ss))
samples = sample(mymodel2(n, b, α), HMC(0.01, 10), 10_000)
nms_rho = filter(startswith("ρ["), string.(samples.name_map.parameters))
samples_rho = samples[nms_rho].value.data#, 10_000, 3)
samples_tau = samples[:τ].data
samples_sd  = 1 ./ sqrt.(k .* samples_tau .* samples_rho)#similar(samples_rho)
mean(samples_sd, dims = 1)

plot(samples_tau)
