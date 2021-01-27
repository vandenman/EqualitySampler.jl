using Plots, Distributions
include("src/newApproach4.jl")


#region replication Figures 1 and 2 of Scott & Berger (2010)
p = 30
D = BetaBinomial(p, 1, 1)
included = 0:p
lpdf = logpdf.(D, included) .- log.(binomial.(p, included))
fig1 = plot(included, lpdf, legend = false);
scatter!(fig1, included, lpdf);
title!(fig1, "Figure 1 of Scott & Berger (2010)");


variables_added = [1, 2, 5, 10]
pos_included = 2:2:100

logpdf_model(D, variable_added) = logpdf(D, variable_added) - log(binomial(ntrials(D), variable_added))
function diff_lpdf(variable_added, p, α::Float64 = 1.0, β::Float64 = 1.0)
	D = BetaBinomial(p, α, β)
	return logpdf_model(D, variable_added - 1) - logpdf_model(D, variable_added)
end

result = Matrix{Float64}(undef, length(variables_added), length(pos_included))
for (i, p) in enumerate(pos_included)
	for j in eachindex(variables_added)
		result[j, i] = diff_lpdf(variables_added[j], p)
	end
end

fig2 = plot(exp.(result)', label = permutedims(["$p included" for p in variables_added]), legend = :topleft);
title!(fig2, "Figure 2 of Scott & Berger (2010)");

plot(fig1, fig2, layout = (2, 1))
#endregion

#region replication with our custom BetaBinomial distribution
k = 15
urns = collect(1:k)
D = BetaBinomialConditionalUrnDistribution(urns, 1, 1, 1)

equalities = 1:k
# first make the figure for small k! also make multivariate distributions!

lpdf = log.(expected_inclusion_probabilities(D)) .- log.(expected_inclusion_counts(k))
fig1 = plot(equalities, lpdf, legend = false);
scatter!(fig1, equalities, lpdf);
title!(fig1, "Log prior probability of a model with x equalities");

function logpdf_model(D, variable_added)
	
	x = 
	logpdf(D, x)
end
function diff_lpdf(variable_added, p, α::Float64 = 1.0, β::Float64 = 1.0)
	D = BetaBinomial(p, α, β)
	return logpdf_model(D, variable_added - 1) - logpdf_model(D, variable_added)
end

result = Matrix{Float64}(undef, length(variable_added), length(pos_included))
for (i, p) in enumerate(pos_included)
	for j in eachindex(variables_added)
		result[j, i] = diff_lpdf(variables_added[j], p)
	end
end

#endregion

expected_inclusion_counts(4)


myStirlings.(15, 5:8) .== Combinatorics.stirlings2.(15, 5:8)


myStirlings(25, 14)
Combinatorics.stirlings2(25, 14)
myStirlings(25, 14)
myStirlings(BigInt(25), BigInt(14))

n = 15
k = 5
nbig = BigInt(n)
kbig = BigInt(k)

@code_warntype myStirlings(nbig, kbig)

@btime stirlings2(n, k)
@btime Combinatorics.stirlings2(n, k)


