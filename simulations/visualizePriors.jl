using EqualitySampler, Plots, Turing
import StatsBase: countmap
include("simulations/samplePriorsTuring.jl")
include("simulations/plotFunctions.jl")
# include("simulations/helperfunctions.jl")

updateDistribution(::UniformConditionalUrnDistribution, urns, j) = UniformConditionalUrnDistribution(urns, j)
updateDistribution(D::BetaBinomialConditionalUrnDistribution, urns, j) = BetaBinomialConditionalUrnDistribution(urns, j, D.α, D.β)

function simulate_from_distribution(nrand, D)
	println("Drawing $nrand draws from $(typeof(D).name)")
	k = length(D)
	urns = copy(D.urns)
	sampled_models = Matrix{Int}(undef, k, nrand)
	for i in 1:nrand
		urns = ones(Int, k)
		for j in 1:k
			D = updateDistribution(D, urns, j)
			urns[j] = rand(D, 1)[1]
		end
		sampled_models[:, i] .= reduce_model(urns)
	end
	return sampled_models
end

k = 4
nrand = 100_000
urns = collect(1:k)
D = UniformConditionalUrnDistribution(urns, 1)
sampled_models = simulate_from_distribution(nrand, D)

empirical_model_probs     = empirical_model_probabilities(sampled_models)
empirical_inclusion_probs = empirical_inclusion_probabilities(sampled_models)

pjoint = visualize_eq_samples(D, empirical_model_probs, empirical_inclusion_probs)
# png(pjoint, "modelspace uniform $k.png")

nrand = 100_000
urns = collect(1:k)
D = BetaBinomialConditionalUrnDistribution(urns, 1, 1, 1)
sampled_models = simulate_from_distribution(nrand, D)

empirical_model_probs     = empirical_model_probabilities(sampled_models)
empirical_inclusion_probs = empirical_inclusion_probabilities(sampled_models)

pjoint = visualize_eq_samples(D, empirical_model_probs, empirical_inclusion_probs)

#region Turing UniformConditionalUrnDistribution
k = 4
uniform_reference_dist = UniformConditionalUrnDistribution(1:k)
equality_prior, empirical_model_probs, empirical_inclusion_probs, _ = sample_and_compute_Turing(k, uniform_prior = true)
visualize_eq_samples(equality_prior, empirical_model_probs, empirical_inclusion_probs)

equality_prior, empirical_model_probs, empirical_inclusion_probs, _ = sample_and_compute_Turing(k, uniform_prior = false)
visualize_eq_samples(equality_prior, empirical_model_probs, empirical_inclusion_probs)

# Note that this prior  puts different weight on 1221 than on 1222
empirical_model_probs, empirical_inclusion_probs, _ = sample_process_turing(k, Turing.RandomMeasures.DirichletProcess(1.0))
visualize_eq_samples(uniform_reference_dist, empirical_model_probs, empirical_inclusion_probs)

empirical_model_probs, empirical_inclusion_probs, _ = sample_process_turing(k, Turing.RandomMeasures.DirichletProcess(3.0))
visualize_eq_samples(uniform_reference_dist, empirical_model_probs, empirical_inclusion_probs)

empirical_model_probs, empirical_inclusion_probs, _ = sample_process_turing(k, Turing.RandomMeasures.PitmanYorProcess(0.5, 0.5, 1))
visualize_eq_samples(uniform_reference_dist, empirical_model_probs, empirical_inclusion_probs)

# using Profile
# using ProfileView

# model = small_model(k, true, 1.0, 1.0)
# samples = sample(model, Prior(), 10_000)
# ProfileView.@profview sample_and_compute_Turing(k, uniform_prior = true, no_samples = 100)

# samples = sample(model, Prior(), 10_000)
# ProfileView.@profview sample(model, Prior(), 10_000)


# model = small_model(k, true, 1.0, 1.0)
# @code_warntype model.f(
#     Random.GLOBAL_RNG,
#     model,
#     Turing.VarInfo(model),
#     Turing.SampleFromPrior(),
#     Turing.DefaultContext(),
#     model.args...,
# )

# m2 = TuringDirichletProcess(k, 1.0)
# @code_warntype m2.f(
#     Random.GLOBAL_RNG,
#     m2,
#     Turing.VarInfo(m2),
#     Turing.SampleFromPrior(),
#     Turing.DefaultContext(),
#     m2.args...,
# )

#endregion

#region Theoretical Plots
function ordering(x)
	# TODO: ensure this is the actual ordering like on wikipedia
	d = countmap(x)
	res = Float64(length(x) - length(d))

	v = sort!(collect(values(d)), lt = !isless)
	# res = 0.0
	for i in eachindex(v)
		res += v[i] ./ 10 .^ (i)
	end
	return res
end

function empty_plot(k; yaxis = :log, legend = true, xrotation = 45, xfontsize = 8, palette = :default)

	allmodels = generate_distinct_models(k)
	order = sortperm(ordering.(eachcol(allmodels)), lt = !isless)
	allmodels_str = [join(col) for col in eachcol(view(allmodels, :, order))]
	transparent = Colors.RGBA(0, 0, 0, 0)
	plt = plot(allmodels_str, ones(length(allmodels_str)), label = nothing, color = transparent,
				yaxis = yaxis, legend = legend, xrotation = xrotation,
				xticks = (0.5:length(allmodels_str)-.5, allmodels_str),
				xtickfont=font(xfontsize),
				tickfontvalign = :bottom,
				palette = palette)
	plt
end

function model_pmf!(plt, D, colors = 1:k, labels = :none)

	k = length(D)
	y = expected_model_probabilities(D)
	incl_size = expected_inclusion_counts(k)
	cumulative_sum = 0
	xstart = 0

	if labels === :none || length(labels) != k
		labs = incl_size
	else
		labs = labels
	end

	for i in 1:k
		yval = y[cumulative_sum + incl_size[i]]
		ycoords = [yval, yval]
		if isone(incl_size[i])
			xcoords = [xstart + 0.5, xstart + 0.5]
			scatter!(plt, xcoords, ycoords, m = 4, label = string(labs[i]), #yaxis = yaxis,
					 color = colors[i], markerstrokecolor = colors[i]);
		else
			xcoords = [xstart, xstart + incl_size[i]]
			plot!(plt, xcoords, ycoords, lw = 4, label = string(labs[i]), #yaxis = yaxis,
				  color = colors[i]);
		end
		cumulative_sum += incl_size[i]
		xstart += incl_size[i]
	end
	return plt
end

function incl_pmf!(plt, D; yaxis = :log, palette = :default, color = 1:length(D), label = fill("", length(D)))
	k = length(D)
	scatter!(plt, 0:k-1, expected_inclusion_probabilities(D), m = 4, yaxis = yaxis, legend = false,
				  color = color, markerstrokecolor = color, palette = palette, label = label);
end

k = 5
Duniform = UniformConditionalUrnDistribution(ones(Int, k))
DBetaBinom = BetaBinomialConditionalUrnDistribution(ones(Int, k), 1, 1, 1)
DDirichletProcess = RandomProcessDistribution(Turing.RandomMeasures.DirichletProcess(1.887), k)

# TODO: for some odd reason the x-axis tick labels are offset...
yaxis_scale = :none
xrotation   = 90
xfontsize   = 4
ylimsModel  = (0, .25)
ylimsIncl   = (0, .5)
colorpalet  = :seaborn_colorblind
palette(colorpalet);

# TODO: this could just be a loop over distributions with parameter vectors
αβ = ((.5, 2), (2, 2), (1, 1), (.5, .5)) # Beta Binomial parameters
αs = (1.887, 1.0, 3.0) # Dirichlet Process parameters
# plt row, column
plt11 = empty_plot(k, legend = false, yaxis = yaxis_scale, xrotation = xrotation, xfontsize = xfontsize, palette = colorpalet);
model_pmf!(plt11, Duniform, fill(1, k))
plot!(plt11, title = "Uniform", ylab = "Probabilty", xlab = "Model", ylims = ylimsModel);

# plt12 = empty_plot(k, legend = false, yaxis = yaxis_scale, xrotation = xrotation, xfontsize = xfontsize, palette = colorpalet);
# model_pmf!(plt12, DBetaBinom);
# plot!(plt12, title = "Beta-binomial (α = $(DBetaBinom.α), β = $(DBetaBinom.β))", xlab = "Model", ylims = ylimsModel);

plt12 = empty_plot(k, legend = :top, yaxis = yaxis_scale, xrotation = xrotation, xfontsize = xfontsize, palette = colorpalet);

for i in eachindex(αβ)
	α, β = αβ[i]
	labels = fill("", k)
	labels[1] = "α = $α, β = $β"
	model_pmf!(plt12, BetaBinomialConditionalUrnDistribution(ones(Int, k), 1, α, β), fill(i, k), labels);
end
plot!(plt12, title = "Beta-binomial", xlab = "Model", ylims = ylimsModel);

# plt12 = empty_plot(k, legend = false, yaxis = yaxis_scale, xrotation = xrotation, xfontsize = xfontsize, palette = colorpalet);
# model_pmf!(plt12, DBetaBinom);
# plot!(plt12, title = "Beta-binomial (α = $(DBetaBinom.α), β = $(DBetaBinom.β))", xlab = "Model", ylims = ylimsModel)

plt13 = empty_plot(k, legend = :top, yaxis = yaxis_scale, xrotation = xrotation, xfontsize = xfontsize, palette = colorpalet);
for i in eachindex(αs)
	α = αs[i]
	labels = fill("", k)
	labels[1] = "α = $α"
	DP = RandomProcessDistribution(Turing.RandomMeasures.DirichletProcess(α), k)
	model_pmf!(plt13, DP, fill(i, k), labels);
end
plot!(plt13, title = "Dirichlet Process", xlab = "Model", ylims = ylimsModel);

# plt13 = empty_plot(k, legend = false, yaxis = yaxis_scale, xrotation = xrotation, xfontsize = xfontsize, palette = colorpalet);
# model_pmf!(plt13, DDirichletProcess);
# plot!(plt13, title = "DirichletProcess (α = $(DDirichletProcess.rpm.α))", xlab = "Model", ylims = ylimsModel);

setAxis!(plt) = plot!(plt, xlab = "No. inequalities", ylab = "Probabilty", ylims = ylimsIncl);

plt21 = plot()
incl_pmf!(plt21, Duniform, yaxis = yaxis_scale, palette = colorpalet, color = fill(1, k));
setAxis!(plt21)

plt22 = plot()
for i in eachindex(αβ)
	α, β = αβ[i]
	labels = fill("", k)
	labels[1] = "α = $α, β = $β"
	incl_pmf!(plt22, BetaBinomialConditionalUrnDistribution(ones(Int, k), 1, α, β), yaxis = yaxis_scale, palette = colorpalet, color = fill(i, k))
end
setAxis!(plt22)

plt23 = plot()
for i in eachindex(αβ)
	α = αs[i]
	labels = fill("", k)
	labels[1] = "α = $α"
	DP = RandomProcessDistribution(Turing.RandomMeasures.DirichletProcess(α), k)
	incl_pmf!(plt23, DP, yaxis = yaxis_scale, palette = colorpalet, color = fill(i, k))
end
setAxis!(plt23)


# plt23 = incl_pmf(DDirichletProcess, yaxis = yaxis_scale, palette = colorpalet);
# plot!(plt23, xlab = "No. inequalities", ylims = ylimsIncl);

w = 420
joint = plot(plt11, plt12, plt13, plt21, plt22, plt23, layout = (2, 3), size = (2w, 3w))
savefig(joint, joinpath("figures", "prior_$k.pdf"))
#endregion
