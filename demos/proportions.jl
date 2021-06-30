using EqualitySampler, Turing, Plots, StatsPlots, FillArrays
import	DataFrames		as DF,
		StatsBase 		as SB,
		LinearAlgebra	as LA,
		NamedArrays		as NA,
		CSV
import DynamicPPL: @submodel

include("simulations/silentGeneratedQuantities.jl")
include("simulations/helpersTuring.jl")

journal_data = DF.DataFrame(CSV.File(joinpath("demos", "data", "journal_data.csv")))

@df journal_data scatter(:journal, :errors, ylims = (0, 1))

@assert journal_data[!, :errors] ≈ journal_data[!, :perc_articles_with_errors] ./ 100

function get_p_constrained(model, samps)

	default_result = generated_quantities2(model, samps)
	clean_result = Matrix{Float64}(undef, length(default_result[1][1][1]), size(default_result, 1))
	for i in eachindex(default_result)
		clean_result[:, i] = default_result[i][1][1]
	end
	return vec(mean(clean_result, dims = 2)), clean_result
end

# TODO: if partition has as default eachindex(no_errors) can we use it instead of proportion_model_full?
@model function proportion_model_inner(no_errors, total_counts, partition)

	p_raw ~ filldist(Beta(1.0, 1.0), length(no_errors))
	p_constrained = p_raw[partition]
	no_errors ~ Distributions.Product(Binomial.(total_counts, p_constrained))
	# for i in eachindex(no_errors)
	# 	no_errors[i] ~ Binomial(total_counts[i], p_constrained[i])
	# end

	return (p_constrained, )
end

@model function proportion_model_full(no_errors, total_counts)
	partition = eachindex(no_errors)
	p = @submodel $(Symbol("inner")) proportion_model_inner(no_errors, total_counts, partition)
	return (p, )
end

@model function proportion_model_eq(no_errors, total_counts, partition_prior)
	partition ~ partition_prior
	p = @submodel $(Symbol("inner")) proportion_model_inner(no_errors, total_counts, partition)
	return (p, )
end

function equality_prob_table(journal_data, samples)
	maxsize = maximum(length, journal_data[!, :journal])
	rawnms = ["$(rpad(journal, maxsize)) ($(round(prob, digits=3)))" for (journal, prob) in eachrow(journal_data[!, [:journal, :errors]])]
	nms = OrderedDict(rawnms .=> axes(journal_data, 1))
	return NA.NamedArray(
		collect(LA.UnitLowerTriangular(compute_post_prob_eq(samps_eq_2))),
		(nms, nms), ("Rows", "Cols")
	)
end

total_counts = journal_data[!, :n]
no_errors = round.(Int, journal_data[!, :x])

model_full = proportion_model_full(no_errors, total_counts)
samps_full = sample(model_full, NUTS(), 50_000);
means_p_full, samps_p_full = get_p_constrained(model_full, samps_full)

p01 = plot(samps_p_full', xlab = "iteration", ylab = "proportions", legend = false, ylim = (0, 1))
p02 = plot();
Plots.abline!(p02, 1, 0, legend = false, color = "black", label = "");
scatter!(p02, journal_data[!, :errors]', means_p_full', legend = :topleft,
		label = permutedims(journal_data[!, :journal]),
		ylim = (0, 1), xlim = (0, 1), xlab = "observed proportion", ylab = "posterior mean");
p1 = plot(p01, p02, layout = (1, 2), size = (800, 400))
# savefig(p1, "demos/proportion_fullmodel.png")

model_eq = proportion_model_eq(no_errors, total_counts, BetaBinomialMvUrnDistribution(length(no_errors), length(no_errors), 1))

# PG does not seem to work well here
# spl_eq_1 = Gibbs(HMCDA(200, 0.65, 0.3, Symbol("inner.p_raw")), PG(5, :partition))
# samps_eq_1 = sample(model_eq, spl_eq_1, 30_000)
# means_p_eq_1, samps_p_eq_1 = get_p_constrained(model_eq, samps_eq_1)

function get_logπ(model)
	vari = VarInfo(model)
	logπ_internal = Turing.Inference.gen_logπ(vari, DynamicPPL.SampleFromPrior(), model)
	return function logπ(partition, c)
		# @show partition, c
		logπ_internal(hcat(Float64.(partition), c[Symbol("inner.p_raw")]))
	end
end

spl_eq_2 = Gibbs(
	# HMCDA seems to get stuck sometimes
	# HMCDA(200, 0.65, 0.3, Symbol("inner.p_raw")),
	HMC(0.05, 10, Symbol("inner.p_raw")),
	GibbsConditional(:partition, EqualitySampler.PartitionSampler(length(no_errors), get_logπ(model_eq)))
)
samps_eq_2 = sample(model_eq, spl_eq_2, 100_000);
means_p_eq_2, samps_p_eq_2 = get_p_constrained(model_eq, samps_eq_2)

# scatter(means_p_eq_1, means_p_eq_2); Plots.abline!(1, 0)
# hcat(get(samps_eq_2, Symbol("inner.p_raw"))[1]...)
plot(hcat(get(samps_eq_2, Symbol("inner.p_raw"))[1]...), labels = reshape(journal_data[!, :journal], 1, 8))

plot(samps_p_eq_2', labels = reshape(journal_data[!, :journal], 1, 8))
scatter(journal_data[!, :errors], means_p_eq_2, legend = false, ylim = (0, 1), xlim = (0, 1)); Plots.abline!(1, 0)

p11 = plot(samps_p_eq_2', xlab = "iteration", ylab = "proportions", legend = false, ylim = (0, 1));
p12 = plot();
Plots.abline!(p12, 1, 0, legend = false, color = "black", label = "")
scatter!(p12, journal_data[!, :errors]', means_p_eq_2', legend = :topleft,
		label = permutedims(journal_data[!, :journal]),
		ylim = (0, 1), xlim = (0, 1), xlab = "observed proportion", ylab = "posterior mean");
p1 = plot(p11, p12, layout = (1, 2), size = (800, 400))
# savefig(p1, "demos/proportion_eqmodel.png")


p12_no_legend = plot(p12, legend = false);
p2 = plot(p02, p12_no_legend, layout = (1, 2), size = (800, 400))
savefig(p2, "demos/proportion_nochains.png")


mp = sort(compute_model_probs(samps_eq_2),  byvalue=true, rev=true)
mc = sort(compute_model_counts(samps_eq_2), byvalue=true, rev=true)
count(!iszero, values(mp))
count(>(0), values(mc)) # number of models visited

equality_prob_table(journal_data, samps_eq_2)



# maxsize = maximum(length, journal_data[!, :journal])
# rawnms = ["$(rpad(journal, maxsize)) ($(round(prob, digits=3)))" for (journal, prob) in eachrow(journal_data[!, [:journal, :errors]])]
# nms = OrderedDict(rawnms .=> axes(journal_data, 1))
# mm = LA.UnitLowerTriangular(compute_post_prob_eq(samps_eq_2))
# NA.NamedArray(mm, (nms, nms), ("Rows", "Cols"))
# NA.NamedArray(rand(8, 8), (nms, nms), ("Rows", "Cols"))
# NA.NamedArray([1 3; 2 4], ( OrderedDict("A"=>1, "B"=>2), OrderedDict("C"=>1, "D"=>2) ), ("Rows", "Cols"))

# function foo(x)
# 	print("c(")
# 	for y in x
# 		print("$y, ")
# 	end
# 	print(")")
# end
# foo(mm)

