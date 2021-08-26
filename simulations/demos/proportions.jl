using EqualitySampler, Turing, Plots, FillArrays, Plots.PlotMeasures
import	DataFrames		as DF,
		StatsBase 		as SB,
		LinearAlgebra	as LA,
		NamedArrays		as NA,
		CSV
import DynamicPPL: @submodel

import Printf
round_2_decimals(x::Number) = Printf.@sprintf "%.2f" x
round_2_decimals(x) = x

include("simulations/silentGeneratedQuantities.jl")
include("simulations/helpersTuring.jl")

journal_data = DF.DataFrame(CSV.File(joinpath("simulations", "demos", "data", "journal_data.csv")))

scatter(journal_data[!, :journal], journal_data[!, :errors], ylims = (0, 1), ylab = "Proportion of statistical reporting errors", label = nothing)

@assert journal_data[!, :errors] ≈ journal_data[!, :perc_articles_with_errors] ./ 100

make_title(d::BetaBinomial) = "Beta-binomial α=$(round_2_decimals(d.α)) β=$(round_2_decimals(d.β))"
make_title(::UniformMvUrnDistribution) = "Uniform"
make_title(d::BetaBinomialMvUrnDistribution) = "Beta-binomial α=$(round_2_decimals(d.α)) β=$(round_2_decimals(d.β))"
make_title(d::RandomProcessMvUrnDistribution) = "Dirichlet Process α=$(round_2_decimals(d.rpm.α))"
make_title(::Nothing) = "Full model"

function get_p_constrained(model, samps)

	default_result = generated_quantities2(model, samps)
	clean_result = Matrix{Float64}(undef, length(default_result[1][1][1]), size(default_result, 1))
	for i in eachindex(default_result)
		clean_result[:, i] = default_result[i][1][1]
	end
	return vec(mean(clean_result, dims = 2)), clean_result
end

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
		collect(LA.UnitLowerTriangular(compute_post_prob_eq(samples))),
		(nms, nms), ("Rows", "Cols")
	)
end

function plot_observed_against_estimated(observed, estimated; legend = :topleft, xlab = "observed proportion", ylab = "posterior mean", kwargs...)
	plt = plot(;kwargs...);
	Plots.abline!(plt, 1, 0, legend = false, color = "black", label = nothing);
	scatter!(plt, observed', estimated', legend = legend, label = permutedims(journal_data[!, :journal]), ylim = (0, 1), xlim = (0, 1), xlab = xlab, ylab = ylab)
	return plt
end

total_counts = journal_data[!, :n]
no_errors = round.(Int, journal_data[!, :x])

priors = (
	full = nothing,
	DPP = DirichletProcessMvUrnDistribution(length(no_errors), 0.5),
	Betabinomial = BetaBinomialMvUrnDistribution(length(no_errors), length(no_errors), 1),
	Uniform = UniformMvUrnDistribution(length(no_errors))
)

function get_logπ(model)
	vari = VarInfo(model)
	logπ_internal = Turing.Inference.gen_logπ(vari, DynamicPPL.SampleFromPrior(), model)
	return function logπ(partition, c)
		# @show partition, c
		logπ_internal(hcat(Float64.(partition), c[Symbol("inner.p_raw")]))
	end
end

fits = map(priors) do prior

	if isnothing(prior)
		model = proportion_model_full(no_errors, total_counts)
		spl = NUTS()
	else
		model = proportion_model_eq(no_errors, total_counts, prior)
		spl = Gibbs(
			HMC(0.05, 10, Symbol("inner.p_raw")),
			GibbsConditional(:partition, EqualitySampler.PartitionSampler(length(no_errors), get_logπ(model)))
		)
	end
	all_samples = sample(model, spl, 100_000);
	# all_samples = sample(model, spl, 5_000);
	posterior_means, posterior_samples = get_p_constrained(model, all_samples)

	return (posterior_means=posterior_means, posterior_samples=posterior_samples, all_samples=all_samples, model=model)
end

trace_plots = map(zip(priors, fits)) do (prior, fit)
	plot(fit[:posterior_samples]', xlab = "iteration", ylab = "proportions", legend = false, ylim = (0, 1), title = make_title(prior))
end
joint_trace_plots = plot(trace_plots..., layout = (1, length(priors)), size = (400length(priors), 400))

retrieval_plots = map(zip(priors, fits)) do (prior, fit)
	plot_observed_against_estimated(journal_data[!, :errors], fit[:posterior_means]; title = make_title(prior),
	legend = isnothing(prior) ? :topleft : nothing, foreground_color_legend = nothing, background_color_legend = nothing)
end
joint_retrieval_plots = plot(retrieval_plots..., layout = (1, length(priors)), size = (400length(priors), 400))
joint_retrieval_plots_2x2 = plot(retrieval_plots..., layout = (2, 2), size = (800, 800))

savefig(joint_trace_plots, "figures/demo_proportions_trace_plots.pdf")
savefig(joint_retrieval_plots, "figures/demo_proportions_retrieval_plots.pdf")
savefig(joint_retrieval_plots_2x2, "figures/demo_proportions_retrieval_plots_2x2.pdf")

# count visited models
visited = map(zip(priors, fits)) do (prior, fit)

	isnothing(prior) && return nothing

	all_samples = fit[:all_samples]
	mp = sort(compute_model_probs(all_samples),  byvalue=true, rev=true)
	mc = sort(compute_model_counts(all_samples), byvalue=true, rev=true)

	return (
		count(!iszero, values(mp)),
		count(>(0), values(mc)), # number of models visited
		equality_prob_table(journal_data, all_samples)
	)

end

visited[4][1]