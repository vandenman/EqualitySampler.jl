using Distributions

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
