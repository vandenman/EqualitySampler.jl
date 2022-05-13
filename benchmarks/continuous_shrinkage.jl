using Distributions, Turing

#=
	TODO:

		- do the Dirichlet stuff analytically!
		- look at https://en.wikipedia.org/wiki/Logit-normal_distribution#Multivariate_generalization

=#
p = 3
ds = Dirichlet.(p:-1:1, 0.1)

pp = zeros(p, p)
for i in axes(pp, 1)
	u = rand(ds[i])
	if i > 1
		u ./= (1.0 / (1.0 - sum(view(pp, 1:i-1, i))))
	end
	pp[i:end, i] .= u
	pp[i, i:end] .= u
end

pp
sum(pp, dims = 1)
sum(pp, dims = 2)

θ = randn(p); θ .-= mean(θ)
pp * θ

DynamicPPL.@model function factor_shrinkage(y, X, g_design, no_g, ::Type{T} = Float64) where {T}

	α 		~ Turing.Flat()
	σ² 		~ EqualitySampler.JeffreysPriorVariance()

	g ~ filldist(Distributions.InverseGamma(0.5, 0.5; check_args = false), no_g)

	G = (g .* g) * g_design

	B ~ prior
	β ~ Distributions.MvNormal(σ² .* G * B)

	μ = α .+ X * β
	y ~ MvNormal(μ, sqrt(σ₂))

end