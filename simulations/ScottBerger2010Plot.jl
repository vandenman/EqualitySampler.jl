using EqualitySampler, Plots, Distributions


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

fig2 = plot(pos_included, exp.(result)', label = permutedims(["$p included" for p in variables_added]), legend = :topleft);
title!(fig2, "Figure 2 of Scott & Berger (2010)");

fig3 = plot(pos_included, result', label = permutedims(["$p included" for p in variables_added]), legend = :topleft);
title!(fig3, "Figure 2 of Scott & Berger (2010) with logarithmic y-axis");

w = 500
plot(fig1, fig2, fig3, size = (w, 2w))
#endregion

#region replication with our custom BetaBinomial distribution
k = 30
urns = collect(1:k)
D = BetaBinomialConditionalUrnDistribution(urns, 1, 1, 1)

equalities = 1:k
# first make the figure for small k! also make multivariate distributions!

lpdf = log.(expected_inclusion_probabilities(D)) .- log.(expected_inclusion_counts(BigInt(k)))
fig1 = plot(equalities, lpdf, legend = false, xlab = "No. inequalities", ylab = "log probability");
scatter!(fig1, equalities, lpdf);
title!(fig1, "prior model probability given the number of inequalities");

variables_added = [1, 2, 5, 10]
pos_included = 2:2:35

DD = BetaBinomialConditionalUrnDistribution(ones(Int, 5), 1, 2.3, 4.6)
BB = BetaBinomial(length(DD) - 1, DD.α, DD.β)
# EqualitySampler._pdf(DD) ./ expected_inclusion_counts(length(DD))
exp.(logpdf_model.(BB, 0:ntrials(BB)))

# NOTE: stirlings2(35, 17) overflows...

function logpdf_model(D, no_inequalities)
	ntrials(D) - no_inequalities + 1 < 0 && return -Inf
	logpdf(D, no_inequalities) - logstirlings2(ntrials(D) + 1, ntrials(D) - no_inequalities + 1)
end

function diff_lpdf(no_inequalities, p, α::Float64 = 1.0, β::Float64 = 1.0)
	D = BetaBinomial(p, α, β)
	return logpdf_model(D, no_inequalities - 1) - logpdf_model(D, no_inequalities)
end

result = Matrix{Float64}(undef, length(variables_added), length(pos_included))
for (i, p) in enumerate(pos_included)
	for j in eachindex(variables_added)
		result[j, i] = diff_lpdf(variables_added[j], p)
	end
end

fig2 = plot(pos_included, result', label = permutedims(["$p included" for p in variables_added]), legend = :topleft);
title!(fig2, "Figure 2 of Scott & Berger (2010)");

plot(fig1, fig2, layout = (2, 1))


#endregion