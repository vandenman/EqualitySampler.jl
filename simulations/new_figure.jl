using EqualitySampler, Distributions
import Random, ProgressMeter
import DataFrames as DF
import StatsBase as SB
import Printf
import ColorSchemes
import JLD2
import LogExpFunctions
using CairoMakie
import AlgebraOfGraphics as AOG
import CairoMakie as CM
import Colors

include("priors_plot_colors_shapes_labels_new.jl")

#region functions
function compute_probs_for_fig(d, current_partition)

    k = length(d)
    @assert k == length(current_partition)
    probvec = zeros(Float64, k)
    partition_sizes = zeros(Int, k)
    if d isa BetaBinomialPartitionDistribution
        EqualitySampler._pdf_helper!(probvec, d, k, current_partition, partition_sizes)

        prob_cl1  = probvec[1]
        prob_cl2  = probvec[2]
        prob_new  = sum(@view probvec[3:end])

    else
        oldval = current_partition[k]
        model_probs = zeros(Float64, 3)
        for j in 1:3
            current_partition[k] = j
            model_probs[j] = pdf_model_distinct(d, current_partition)
        end
        current_partition[k] = oldval

        model_probs ./= sum(model_probs)

        prob_cl1  = model_probs[1]
        prob_cl2  = model_probs[2]
        prob_new  = model_probs[3]

    end


    return prob_cl1, prob_cl2, prob_new
end

create_partition(k::T) where T<:Integer = T[fill(1, k - 2); 2; 0]


function conditional_probs_last_θ(d, partition, max_k = length(d) - 1)

    k = length(d)
    @assert k == length(partition)
    @show max_k
    uk = EqualitySampler.no_distinct_groups_in_partition(view(partition, 1:max_k))
    model_probs = zeros(Float64, uk + 1)

    old_value = partition[k]
    for j in eachindex(model_probs)
        partition[k] = j
        model_probs[j] = pdf_model_distinct(d, partition)
    end
    partition[k] = old_value

    return model_probs ./= sum(model_probs)
end
function rand_θ_new(d, partition, base_measure = Normal())
    cond_probs = conditional_probs_last_θ(d, partition)
    idx = rand(Categorical(cond_probs))
    θ = if idx == 1
        rand(base_measure)
    elseif idx == 2
        rand(base_measure)
    else
        rand(base_measure)
    end
    return θ
end

function conditional_probs_last_two_θ(d, partition)

    k = length(d)
    @assert k == length(partition)

    uk = EqualitySampler.no_distinct_groups_in_partition(view(partition, 1:k - 2))

    no_models = uk * (uk + 1) + (uk + 2)
    models      = Vector{Vector{Int}}(undef, no_models)
    model_probs = Vector{Float64}(undef, no_models)

    idx = 1
    old_value_k_j_min_1, old_value_k_j = partition[k-1:k]
    for k_j_min_1 in 1:uk + 1

        partition[k-1] = k_j_min_1
        uk_k = uk + 1 + (k_j_min_1 == uk + 1)
        for k_j in 1:uk_k
            partition[k] = k_j
            models[idx] = copy(partition)
            model_probs[idx] = pdf_model_distinct(d, partition)
            # @show k_j, k_j_min_1, idx
            idx += 1
        end
    end
    partition[k-1:k] .= old_value_k_j_min_1, old_value_k_j

    return model_probs ./ sum(model_probs), models

end

function rand_θ_joint(n, d, partition, base_measure = Normal())

    k = length(d)
    cond_probs, models = conditional_probs_last_two_θ(d, partition)

    vals = Matrix{Float64}(undef, n, 2)
    for i in 1:n

        idx = rand(Categorical(cond_probs))
        m = models[idx]

        uk = maximum(m)
        if m[k - 1] == m[k]
            val = rand(base_measure)
            vals[i, 1] = val
            vals[i, 2] = val
        else
            vals[i, 1] = rand(base_measure)
            vals[i, 2] = rand(base_measure)
        end
    end
    return vals
end

harmonic_number(n) = sum(1 / i for i in n:-1:1)


function sample_one_equals_two(d, no_samples = 10_000)

    sample = rand(d)
    result = Int(sample[1] == sample[2])
    for _ in 2:no_samples
        Distributions.rand!(d, sample)
        result += Int(sample[1] == sample[2])
    end
    return result / no_samples

end

compute_one_equals_two(d::DirichletProcessPartitionDistribution) = 1 / (1 + d.α)


function compute_one_equals_two(d::Union{BetaBinomialPartitionDistribution, UniformPartitionDistribution, PitmanYorProcessPartitionDistribution})
    probvec = zeros(Float64, length(d))
    x = zeros(Int, length(d))
    x[1] = 1
    partition_size = similar(x)
    EqualitySampler._pdf_helper!(probvec, d, 2, x, partition_size)
    return probvec[1]
end

# d_dp = DirichletProcessPartitionDistribution(5, 0.3)
# v0 = sample_one_equals_two(d_dp, 1_000_000)
# v1 = compute_one_equals_two(d_dp)
# v0, v1, v0 - v1

# d_bb = BetaBinomialPartitionDistribution(5, 4.20, 0.46)
# w0 = sample_one_equals_two(d_bb, 1_000_000)
# w1 = compute_one_equals_two(d_bb)
# w0, w1, w0 - w1

# d_u = UniformPartitionDistribution(5)
# u0 = sample_one_equals_two(d_u, 1_000_000)
# u1 = compute_one_equals_two(d_u)
# u0, u1, u0 - u1

# d_py = PitmanYorProcessPartitionDistribution(5, 0.45, -0.2)
# v0 = sample_one_equals_two(d_py, 1_000_000)
# v1 = compute_one_equals_two(d_py)
# v0, v1, v0 - v1

# compute_one_equals_two(d)

#=
# TODO: remove these in favor of "prediction_rule", perhaps export that?
function equation_12(k, r, α, β)

    # TODO: assumes we're sampling from the last urn, so j = k-1?
    # double check this with the result from _pdf_helper!

    d = BetaBinomialPartitionDistribution(k, α, β)
    model_probs_by_incl = exp.(EqualitySampler.log_model_probs_by_incl(d))

    index = k# - 1
    n = k - (index - r - 1)

    num =     sum(model_probs_by_incl[i] * stirlings2r(n    , i, r + 1) for i in 1:k)
    den = r * sum(model_probs_by_incl[i] * stirlings2r(n - 1, i, r    ) for i in 1:k)
    return num / (num + den)

end

function prediction_rule_DP(k, r, α)
    α / (k + α - 1)
end

function prediction_rule_PY(k, r, θ, α)
    (θ + α * r) / (θ + k - 1)
end

function prediction_rule(d::AbstractPartitionDistribution, r::Integer)

    k = length(d)
    @assert 1 <= r <= k
    partition = ones(typeof(k), k)
    partition[k-r+1:k-1] .= 2:r
    probvec = zeros(k)
    EqualitySampler._pdf_helper!(probvec, d, k, partition, zeros(Int, k))
    return sum(probvec[eachindex(probvec) .∉ Ref(1:r)])#view(partition, 1:k-1))])

end

prediction_rule(d::Type{<:AbstractPartitionDistribution}, r::Integer, args...) = prediction_rule(d(args...), r)

# prediction_rule(BetaBinomialPartitionDistribution(4, .1, .2), 2)
# prediction_rule(BetaBinomialPartitionDistribution, 2, 4, .1, .2)

# k = 4; r = 2; partition = [1, 1, 2, -1]
# d = PitmanYorProcessPartitionDistribution(k, 0.3, 0.5)
# probvec = zeros(length(d))
# EqualitySampler._pdf_helper!(probvec, d, k, partition, zeros(Int, k))
# prediction_rule_PY(length(d), r, d.θ, d.d) ≈ sum(probvec[eachindex(probvec) .∉ Ref([1, 1, 2])]) ≈
#   prediction_rule(d, 2)

# dpp = DirichletProcessPartitionDistribution(k, .123)
# EqualitySampler._pdf_helper!(probvec, dpp, k, partition, zeros(Int, 4))
# prediction_rule_DP(k, 3, dpp.α) ≈ sum(probvec[eachindex(probvec) .∉ Ref([1, 1, 2])]) ≈
#   prediction_rule(dpp, 2)

# dbb = BetaBinomialPartitionDistribution(k, .123, 6.45)
# EqualitySampler._pdf_helper!(probvec, dbb, k, partition, zeros(Int, 4))
# equation_12(k, r, dbb.α, dbb.β) ≈ sum(probvec[eachindex(probvec) .∉ Ref([1, 1, 2])]) ≈
#   prediction_rule(dbb, 2)

=#
#endregion

# models = (
#     (:bb_1,       k -> BetaBinomialPartitionDistribution(k, 1, 1)),
#     (:bb_k,       k -> BetaBinomialPartitionDistribution(k, 1, k)),
#     (:bb_binom1k, k -> BetaBinomialPartitionDistribution(k, 1, max(1, binomial(k, 2)))),
#     # (:bb_binomk1, k -> BetaBinomialPartitionDistribution(k, binomial(k, 2), 1)),

#     (:dp_1,     k -> DirichletProcessPartitionDistribution(k, 1.0)),
#     # (:dp_0_5,   k -> DirichletProcessPartitionDistribution(k, 0.5)),
#     # (:dp_2,     k -> DirichletProcessPartitionDistribution(k, 2.0)),
#     (:dp_GB,    k -> DirichletProcessPartitionDistribution(k, isone(k) ? 1 : :Gopalan_Berry)),
#     (:dp_decr,  k -> DirichletProcessPartitionDistribution(k, isone(k) ? 1 : 1 / harmonic_number(k-1))),

#     # (:py_0_5_0_5,    k -> PitmanYorProcessPartitionDistribution(k, 0.5, 0.5)),
#     # (:py_0_5_1_0,    k -> PitmanYorProcessPartitionDistribution(k, 0.5, 1.0)),
#     # (:py_0_5_decr,   k -> PitmanYorProcessPartitionDistribution(k, 0.5, isone(k) ? 1 : 1 / harmonic_number(k-1))),
#     # (:py_1_0_decr,   k -> PitmanYorProcessPartitionDistribution(k, 0.9, isone(k) ? 1 : 1 / harmonic_number(k-1))),
#     # (:py_1_0_1_0,    k -> PitmanYorProcessPartitionDistribution(k, 0.2, 1.0)),

#     (:uniform,  k -> UniformPartitionDistribution(k))
# )


models = (
    (:BetaBinomial11,       k -> BetaBinomialPartitionDistribution(k, 1, 1)),
    # (:BetaBinomialk1,       k -> BetaBinomialPartitionDistribution(k, 1, k)),
    (:BetaBinomial1k,       k -> BetaBinomialPartitionDistribution(k, 1, k)),
    (:BetaBinomial1binomk2, k -> BetaBinomialPartitionDistribution(k, 1, binomial(k, 2))),


    # (:DirichletProcess0_5,  k -> DirichletProcessPartitionDistribution(k, 0.5)),
    (:DirichletProcess1_0,  k -> DirichletProcessPartitionDistribution(k, 1.0)),
    # (:DirichletProcess2_0,  k -> DirichletProcessPartitionDistribution(k, 2.0)),
    (:DirichletProcessGP,   k -> DirichletProcessPartitionDistribution(k, :Gopalan_Berry)),
    (:DirichletProcessDecr, k -> DirichletProcessPartitionDistribution(k, :harmonic)),

    # (:PitmanYorProcess0_5__0_5,    k -> PitmanYorProcessPartitionDistribution(k, 0.5, 0.5)),
    # (:PitmanYorProcess0_5__1_0,    k -> PitmanYorProcessPartitionDistribution(k, 0.5, 1.0)),
    # (:PitmanYorProcess0_5__decr,   k -> PitmanYorProcessPartitionDistribution(k, 0.5, isone(k) ? 1 : 1 / harmonic_number(k-1))),
    # (:PitmanYorProcess1_0__decr,   k -> PitmanYorProcessPartitionDistribution(k, 0.9, isone(k) ? 1 : 1 / harmonic_number(k-1))),
    # (:PitmanYorProcess1_0__1_0,    k -> PitmanYorProcessPartitionDistribution(k, 0.2, 1.0)),

    (:uniform,  k -> UniformPartitionDistribution(k))
)
prior_nms = collect(string.(first.(models)))

ks = 3:30

iter = Iterators.product(ks, models)

nresults = length(iter)
results_df = DF.DataFrame(
    model                = Vector{String}( undef, nresults),
    k                    = Vector{Int}(    undef, nresults),
    prob_big             = Vector{Float64}(undef, nresults),
    prob_small           = Vector{Float64}(undef, nresults),
    prob_new             = Vector{Float64}(undef, nresults),
    prob_null            = Vector{Float64}(undef, nresults),
    prob_one_equals_two  = Vector{Float64}(undef, nresults),
    expected_no_clusters = Vector{Float64}(undef, nresults),
    median_no_clusters   = Vector{Int}(undef, nresults)
)

ProgressMeter.@showprogress for (row_idx, (k, (modelname, modelfun))) in enumerate(iter)

    d = modelfun(k)
    partition = create_partition(k)
    res = compute_probs_for_fig(d, partition)

    results_df.model[row_idx]      = string(modelname)
    results_df.k[row_idx]          = k
    results_df.prob_big[row_idx]   = res[1]
    results_df.prob_small[row_idx] = res[2]
    results_df.prob_new[row_idx]   = res[3]
    results_df.prob_null[row_idx]  = logpdf_model_distinct(d, ones(Int, k))

    # results_df.prob_one_equals_two[row_idx] = sample_one_equals_two(d)
    results_df.prob_one_equals_two[row_idx] = compute_one_equals_two(d)

    # @show k, modelname
    d_cat = Distributions.Categorical(pdf_incl.(Ref(d), 1:k))
    results_df.expected_no_clusters[row_idx] = mean(d_cat)
    results_df.median_no_clusters[row_idx]   = median(d_cat)

end

results_df.prob_big_normalized = results_df.prob_big ./ (results_df.prob_big .+ results_df.prob_small)

# prediction_rule_df = DF.DataFrame(
#     model        = collect(string.(first.(models))),
#     vals_heatmap = [fill(NaN, maximum(ks), maximum(ks)) for _ in eachindex(models)]
# )

# ProgressMeter.@showprogress for (row_idx, (modelname, modelfun)) in enumerate(models)
#     for k in 1:maximum(ks)
#         d = modelfun(k)
#         for r in 1:k-1
#             prediction_rule_df.vals_heatmap[row_idx][k, r] = prediction_rule(d, r)
#         end
#     end
# end

# prediction_rule_df_long = DF.DataFrame(
#     model        = repeat(collect(string.(first.(models))), inner = maximum(ks)^2),
#     k            = repeat(1:maximum(ks), length(models) * maximum(ks)),
#     r            = repeat(1:maximum(ks), inner = maximum(ks), outer = length(models)),
#     vals_heatmap = mapreduce(vec, vcat, prediction_rule_df.vals_heatmap)
# )

# fig = (AOG.data(prediction_rule_df_long) *
#     AOG.mapping(:k, :r, :vals_heatmap, layout = :model) *
#     AOG.visual(CM.Heatmap)
# ) |> AOG.draw()
# CM.save("simulations/newfigures/new_figure_allprediction_rule_heatmaps.pdf", fig)

# (AOG.data(prediction_rule_df_long) *
#     AOG.mapping(:k, :r, :vals_heatmap, layout = :model) *
#     AOG.visual(CM.Heatmap; colorrange = (0, 1))
# ) |> AOG.draw()

# colors_uniform = :grey
# colors_bb      = :managua10
# colors_dpp     = :bamako10
# colors_py      = :lajolla10
# bb_idx = 1
# dpp_idx = 1
# py_idx = 1

# all_linecolors_dict = Dict{String, Colors.ARGB{Float64}}()
# for prior in prior_nms
#     if prior == "uniform" all_linecolors_dict[prior] = Colors.GrayA(.5)
#     elseif startswith(prior, "bb")
#         all_linecolors_dict[prior] = reverse(getfield(ColorSchemes, colors_bb))[bb_idx]
#         bb_idx += 2
#     elseif startswith(prior, "dp")
#         all_linecolors_dict[prior] = getfield(ColorSchemes, colors_dpp)[dpp_idx]
#         dpp_idx += 2
#     elseif startswith(prior, "py")
#         all_linecolors_dict[prior] = getfield(ColorSchemes, colors_py)[py_idx]
#         py_idx += 2
#     else
#         throw(ArgumentError("Unknown prior: $prior"))
#     end
# end
# fig = CM.Figure()
# bb_idx, dpp_idx, py_idx = 1, 1, 1
# Label(fig[0, 1], "Uniform")
# Label(fig[0, 2], "Betabinomial")
# Label(fig[0, 3], "Dirichlet")
# Label(fig[0, 4], "Pitman Yor")
# for prior in prior_nms
#     if prior == "uniform"
#         CM.Box(fig[1, 1], color = all_linecolors_dict[prior])
#     elseif startswith(prior, "bb")
#         CM.Box(fig[bb_idx, 2], color = all_linecolors_dict[prior])
#         bb_idx += 1
#     elseif startswith(prior, "dp")
#         CM.Box(fig[dpp_idx, 3], color = all_linecolors_dict[prior])
#         dpp_idx += 1
#     elseif startswith(prior, "py")
#         CM.Box(fig[py_idx, 4], color = all_linecolors_dict[prior])
#         py_idx += 1
#     else
#         throw(ArgumentError("Unknown prior: $prior"))
#     end
# end
# fig

# all_linecolors = ColorSchemes.alphacolor.(ColorSchemes.seaborn_colorblind[1:10], 0.75)
# all_linecolors_dict = Dict(
#     "uniform"             => all_linecolors[1],
#     "bb_1"                => all_linecolors[2],
#     # :BetaBinomialk1       => colors[3],
#     "bb_k"                => all_linecolors[4],
#     "bb_binom1k"          => all_linecolors[5],
#     "dp_0_5"              => all_linecolors[6],
#     "dp_1"                => all_linecolors[7],
#     "dp_2"                => all_linecolors[8],
#     "dp_GB"               => all_linecolors[9],
#     "dp_decr"             => all_linecolors[3],

#     "py_0_5_0_5"          => all_linecolors[8],
#     "py_1_1"              => all_linecolors[6],
#     "py_0_5_1_0"          => all_linecolors[1],
#     # :Westfall             => colors[10],
#     # :Westfall_uncorrected => colors[3]
# )
function linestyle_mapper(prior)
    if prior == "uniform" return :dashdot
    elseif startswith(prior, "Beta") return :solid
    elseif startswith(prior, "Diri") return :dash
    elseif startswith(prior, "py") return :dashdot
    end
    throw(ArgumentError("Unknown prior: $prior"))
end
linestyle_dict = Dict(Symbol(prior) => linestyle_mapper(prior) for prior in prior_nms)

# linesize_dict = Dict(
#     "uniform"             => 2.0,
#     "bb_1"                => 2.0,
#     "bb_k"                => 2.0,
#     "bb_binom1k"          => 2.0,
#     "dp_0_5"              => 2.0,
#     "dp_1"                => 2.0,
#     "dp_2"                => 2.0,
#     "dp_GB"               => 2.0,
#     "dp_decr"             => 2.0,
#     "py_0_5_0_5"          => 2.0,
#     "py_1_1"              => 2.0,
#     "py_0_5_1_0"          => 2.0,
# )

results_df.model_sym = Symbol.(results_df.model)
color_palette  = get_color_palette(unique(results_df.model_sym))
marker_palette = get_marker_palette(unique(results_df.model_sym))

legend_elems, legend_contents, legend_titles = get_legend_contents(color_palette, nothing, true, linestyle_dict, 3)


results_df.prob_null_exp = exp.(results_df.prob_null)
# priors_to_keep = filter(!startswith("py_"), unique(results_df.model))
aog_data = AOG.data(results_df)#DF.subset(results_df, :model => x -> in.(x, Ref(priors_to_keep))))
fig1 = aog_data *
    AOG.mapping(:k, :prob_null_exp, linestyle = :model_sym, color = :model_sym) *
    AOG.visual(CM.Lines, linewidth = 4, alpha = .8)
fig2 = aog_data *
    AOG.mapping(:k, :prob_one_equals_two, linestyle = :model_sym, color = :model_sym) *
    AOG.visual(CM.Lines, linewidth = 4, alpha = .8)
fig3 = aog_data *
    AOG.mapping(:k, :expected_no_clusters, linestyle = :model_sym, color = :model_sym) *
    AOG.visual(CM.Lines, linewidth = 4, alpha = .8)


color_scale     = (; palette = collect(pairs(color_palette)))
linestyle_scale = (; palette = collect(pairs(linestyle_dict)))

xticks = [3, 5, 10, 15, 20, 25, 30]
yticks = [0, 0.25, 0.5, 0.75, 1]
lims = (2.9, 30.1, 0.00, 1.02)
axis_args_no_yticks = (xlabel = "K", xticks = xticks, rightspinevisible = false, topspinevisible = false)
axis_args = (axis_args_no_yticks..., limits = lims, yticks = yticks)

# AOG.draw(fig11, AOG.scales(Color = color_scale, LineStyle = linestyle_scale), axis = (title = "Probability of a new cluster", axis_args...))
# AOG.draw(fig21, AOG.scales(Color = color_scale, LineStyle = linestyle_scale), axis = (title = "Probability of joining the largest cluster", axis_args...))
# AOG.draw(fig12, AOG.scales(Color = color_scale, LineStyle = linestyle_scale), axis = (title = "Probability of joining the largest cluster", axis_args...))
# AOG.draw(fig22, AOG.scales(Color = color_scale, LineStyle = linestyle_scale), axis = (title = "Expected number of clusters", limits = ((lims[1], lims[2]), nothing), axis_args_no_yticks...))


fig = Figure(fontsize = 20)
gl1 = fig[1, 1] = GridLayout()
ax1 = Axis(gl1[1, 1], title = "Probability of the null model"; ylabel = "Probability", axis_args...)
ax2 = Axis(gl1[1, 2], title = "Probability θᵢ = θⱼ"; axis_args...)
ax3 = Axis(gl1[2, 1], title = "Expected number of clusters", limits = ((lims[1], lims[2]), nothing); axis_args_no_yticks...)
AOG.draw!(ax1, fig1, AOG.scales(Color = color_scale, LineStyle = linestyle_scale))
AOG.draw!(ax2, fig2, AOG.scales(Color = color_scale, LineStyle = linestyle_scale))
AOG.draw!(ax3, fig3, AOG.scales(Color = color_scale, LineStyle = linestyle_scale))

ord = [2, 3, 1]
CM.Legend(gl1[2, 2], legend_elems[ord], legend_contents[ord], legend_titles[ord], 
    tellwidth = false, tellheight = false,
    halign = :center, valign = :center, titlehalign = :left,
    labelhalign = :left,
    gridshalign = :left,
    position = :ct, framevisible = false, backgroundcolor = :transparent, margin = (0, 0, 0, -5))

# make_legend!(gl1, legend_elems, legend_contents, legend_titles, 2, 2, 2:3)

w = 650
CM.resize!(fig, 2w, 800)
fig

figures_dir = joinpath(pwd(), "simulations", "revision_figures")
save(joinpath(figures_dir, "new_figure_1x3_2.pdf"), fig)


# TODO: group per prior
legend_elements = [
    [
        CM.LineElement(
            color     = all_linecolors_dict[prior],
            linestyle = linestyle_dict[prior],
            # linewidth = 1.6
        )
    ]
    for prior in priors_to_keep
]
legend_prior_names_dct = Dict(
    "bb_1"       => "BetaBinomial(1, 1)",
    "bb_k"       => "BetaBinomial(1, k)",
    # "bb_binom1k" => L"\mathrm{BetaBinomial}(1, k \choose 2)",
    "bb_binom1k" => "BetaBinomial(1, binomial(k, 2))",
    "dp_1"       => "DirichletProcess(1)",
    "dp_GB"      => "DirichletProcess(GB)",
    "dp_decr"    => "DirichletProcess(1 / H(k-1))",
    "uniform"    => "Uniform"
)
legend_prior_names = [legend_prior_names_dct[prior] for prior in priors_to_keep]
legend_prior_idx   = [1:3, 4:6, 7:7]

legend_elements2    = [legend_elements[i] for i in legend_prior_idx]
legend_prior_names2 = [legend_prior_names[i] for i in legend_prior_idx]
legend_prior_titles = ["BetaBinomial", "DirichletProcess", "Uniform"]
legend_prior_names2 = [
    ["α = 1, β = 1", "α = 1, β = k", "α = 1, β = binom(k, 2)"],
    ["α = 1", "α = Gopalan & Berry", "α = 1 / H(k-1)"],
    ["Uniform"]
]
Legend(gl1[3, :], legend_elements2, legend_prior_names2, legend_prior_titles, orientation = :horizontal, framevisible = false, nbanks = 3)
fig

w = 400
CM.resize!(fig, w*3, ceil(Int, w * 1))
fig

save("simulations/newfigures/new_figure_1x3_2.pdf", fig)





fig11 = AOG.data(results_df) *
    AOG.mapping(:k, :prob_new, linestyle = :model, color = :model) *
    AOG.visual(CM.Lines, linewidth = 2)
fig21 = AOG.data(results_df) *
    AOG.mapping(:k, :prob_big, linestyle = :model, color = :model) *
    AOG.visual(CM.Lines, linewidth = 2)

fig12 = AOG.data(results_df) *
    AOG.mapping(:k, :prob_big_normalized, linestyle = :model, color = :model) *
    AOG.visual(CM.Lines, linewidth = 2)

fig22 = AOG.data(results_df) *
    AOG.mapping(:k, :expected_no_clusters, linestyle = :model, color = :model) *
    AOG.visual(CM.Lines, linewidth = 2)


color_scale     = (; palette = collect(pairs(all_linecolors_dict)))
linestyle_scale = (; palette = collect(pairs(linestyle_dict)))

xticks = [1, 5, 10, 15, 20, 25, 30]
yticks = [0, 0.25, 0.5, 0.75, 1]
lims = (1, 30, -0.02, 1.02)
axis_args_no_yticks = (xticks = xticks, rightspinevisible = false, topspinevisible = false)
axis_args = (axis_args_no_yticks..., limits = lims, yticks = yticks)

AOG.draw(fig11, AOG.scales(Color = color_scale, LineStyle = linestyle_scale), axis = (title = "Probability of a new cluster", axis_args...))
AOG.draw(fig21, AOG.scales(Color = color_scale, LineStyle = linestyle_scale), axis = (title = "Probability of joining the largest cluster", axis_args...))
AOG.draw(fig12, AOG.scales(Color = color_scale, LineStyle = linestyle_scale), axis = (title = "Probability of joining the largest cluster", axis_args...))
AOG.draw(fig22, AOG.scales(Color = color_scale, LineStyle = linestyle_scale), axis = (title = "Expected number of clusters", limits = ((lims[1], lims[2]), nothing), axis_args_no_yticks...))


fig = Figure(size = (3, 2) .* 400)
gl1 = fig[1, 1] = GridLayout()
ax11 = Axis(gl1[1, 1], title = "Probability of a new cluster"; axis_args...)
ax21 = Axis(gl1[2, 1], title = "Probability of joining the largest cluster"; axis_args...)
ax12 = Axis(gl1[1, 2], title = "Probability of joining the largest cluster"; axis_args...)
ax22 = Axis(gl1[2, 2], title = "Expected number of clusters", limits = ((lims[1], lims[2]), nothing); axis_args_no_yticks...)
AOG.draw!(ax11, fig11, AOG.scales(Color = color_scale, LineStyle = linestyle_scale))
AOG.draw!(ax21, fig21, AOG.scales(Color = color_scale, LineStyle = linestyle_scale))
AOG.draw!(ax12, fig12, AOG.scales(Color = color_scale, LineStyle = linestyle_scale))
AOG.draw!(ax22, fig22, AOG.scales(Color = color_scale, LineStyle = linestyle_scale))

Label(gl1[3, :], "K")

prior_nms = collect(string.(first.(models)))
legend_elements = [
    [
        CM.LineElement(
            color     = all_linecolors_dict[prior],
            linestyle = linestyle_dict[prior],
        )#,
        # CM.MarkerElement(
        #     color     = all_linecolors_dict[prior],
        #     linestyle = linestyle_dict[prior],
        # )
    ]
    for prior in prior_nms
]
Legend(gl1[4, :], legend_elements, prior_nms, orientation = :horizontal)
fig

gl2 = fig[2, 1] = GridLayout()

ax31 = Axis(gl2[1, 1], title = "Probability of a new cluster given r. BB(1, 1) / BB(1, k)",      xticks = xticks, yticks = xticks)
ax32 = Axis(gl2[1, 2], title = "Probability of a new cluster given r. PY(1/2, 1/2) / PY(1, 1) ", xticks = xticks, yticks = xticks)
heatmap_values = [
    i == j ? NaN :
        i <  j ? prediction_rule_df.vals_heatmap[1][j, i] :
                 prediction_rule_df.vals_heatmap[2][i, j]
    for i in axes(prediction_rule_df.vals_heatmap[1], 1), j in axes(prediction_rule_df.vals_heatmap[1], 2)
]
CM.heatmap!(ax31, heatmap_values, colorrange = (0, 1))



heatmap_values = [
    i == j ? NaN :
        i <  j ? prediction_rule_df.vals_heatmap[7][j, i] :
                 prediction_rule_df.vals_heatmap[8][i, j]
    for i in axes(prediction_rule_df.vals_heatmap[1], 1), j in axes(prediction_rule_df.vals_heatmap[1], 2)
]
CM.heatmap!(ax32, heatmap_values, colorrange = (0, 1))

fig

w = 480
CM.resize!(fig, w*2, ceil(Int, w * 3.2))
save("simulations/newfigures/new_figure_3x2.pdf", fig)

CM.heatmap(prediction_rule_df.vals_heatmap[1])
CM.heatmap(prediction_rule_df.vals_heatmap[2])


ax2 = Axis(fig[2, 1], title = "Probability of joining the largest cluster"; axis_args...)



linecolors = [all_linecolors_dict[m] for m in results_df.model]
linestyle  = [linestyle_dict[m] for m in results_df.model]
# linesize  = [linesize_dict[m] for m in results_df.model]
linesize  = [2 for m in results_df.model]

results_df_grouped = DF.groupby(results_df, :model)


fig = Figure(size = (3, 2) .* 400)
xticks = [1, 5, 10, 15, 20, 25, 30]
yticks = [0, 0.25, 0.5, 0.75, 1]
# ylims = (0, 1)
# xlims = (1, 30)
lims = (1, 30, -0.02, 1.02)
ax1 = Axis(fig[1, 1], xticks = xticks, yticks = yticks, limits = lims, rightspinevisible = false, topspinevisible = false,
    title = "Probability of a new cluster")

# (i, df) = first(enumerate(results_df_grouped))
obj_for_legend = []
nms_for_legend = String[]
for (i, df) in enumerate(results_df_grouped)
    model = df.model[1]
    ll = lines!(ax1, ks, df.prob_new, label = model, linestyle = linestyle_dict[model], color = all_linecolors_dict[model], linewidth = 2)
    push!(obj_for_legend, ll)
    push!(nms_for_legend, model)
end
# axislegend(ax1, position = (:right, :top), nbanks = 2)
Legend(fig[0, 1:2], obj_for_legend, nms_for_legend, nbanks = length(nms_for_legend))# ÷ 2)#, nbanks = length(obj_for_legend))
fig



ax2 = Axis(fig[2, 1], xticks = xticks, yticks = yticks, limits = lims, rightspinevisible = false, topspinevisible = false,
    title = "Probability of joining the largest cluster")
# (i, df) = first(enumerate(results_df_grouped))
for (i, df) in enumerate(results_df_grouped)
    model = df.model[1]
    lines!(ax2, ks, df.prob_big, label = model, linestyle = linestyle_dict[model], color = all_linecolors_dict[model], linewidth = 2)
end
fig

# linestyle = repeat(reduce(vcat, fill.([:solid, :dash, :dashdot], [3, 5, 1])), inner = length(ks))
# linesize  = repeat(reduce(vcat, fill.([2.0, 2.0, 2.0], [3, 5, 1])), inner = length(ks))

# p1 = plot(repeat(ks, length(models)), results_df.prob_new, group = results_df.model, legend = :topright,
#     title = "Probability of a new cluster", ylim = (0, 1), xlab = "K", linestyle = linestyle, linewidth = linesize, linecolor = linecolors)
# p2 = plot(repeat(ks, length(models)), results_df.prob_big, group = results_df.model, legend = false,
#     title = "Probability of joining the largest cluster", ylim = (0, 1), xlab = "K", linestyle = linestyle, linewidth = linesize, linecolor = linecolors)

# p3 = plot(repeat(ks, length(models)), exp.(results_df.prob_null), group = results_df.model, legend = false,
#     title = "Probability of the null model", ylim = (0, 1), xlab = "K", linestyle = linestyle, linewidth = linesize)

# plot(p1, p2)

αβ_bb = (:one_k, )#:uniform, )
α_dp  = (1.0, )
# αθ_py = ((1.0, 0.0), (1.0, 0.2), (1.0, 0.5), (1.0, 0.9), (0.1, 1.0))
αθ_py = [(1.0, i) for i in 0.0:0.1:1.0]#, (1.0, 0.2), (1.0, 0.5), (1.0, 0.9), (0.1, 1.0))
max_k = 30

vals_heatmap_bb = fill(NaN, max_k, max_k)
vals_heatmap_dp = fill(NaN, max_k, max_k)

vals_heatmap_py = fill(NaN, max_k, max_k, length(αθ_py))


prog = ProgressMeter.Progress((max_k * (max_k - 1)) ÷ 2, showspeed=true)
for k in axes(vals_heatmap_bb, 1)
    for r in 1:k
        if r != k
            for (j, method) in enumerate(αβ_bb)

                α, β = if method == :uniform
                    1., 1.
                elseif method == :one_k
                    1., float(k)
                else
                    1., max(1, binomial(k, 2))
                end
                if k > 30
                    vals_heatmap_bb[k, r] = Float64(equation_12(big(k), big(r), α, β))
                else
                    vals_heatmap_bb[k, r] = equation_12(k, r, α, β)
                end
            end
            for (j, method) in enumerate(α_dp)

                α = if j <= 3
                    method
                elseif j == 4
                    EqualitySampler.dpp_find_α(k)
                else
                    isone(k) ? 1.0 : 1 / harmonic_number(k - 1)
                end
                vals_heatmap_dp[k, r] = α / (α + k - 1)

            end
            for (j, method) in enumerate(αθ_py)
                θ, β = method
                vals_heatmap_py[k, r, j] = prediction_rule_PY(k, r, θ, β)
            end
        end
        ProgressMeter.next!(prog)
    end
end

equation_12(3, 1, 1, 3)
equation_12(3, 3, 1, 3)


CairoMakie.activate!(inline = true)
fig = Figure(size = (4, 4) .* 400)
extrema(Iterators.filter(!isnan, vals_heatmap_py))
clim_max = 1.0
# [maximum(Iterators.filter(!isnan, vals_heatmap_py[:, :, i])) for i in axes(vals_heatmap_py, 3)]
for i in axes(vals_heatmap_py, 3)
    i1 = mod1(i, 3)
    i2 = div(i - 1, 3) + 1
    ax = Axis(fig[i1, i2], title = "α = $(αθ_py[i][1]), θ = $(αθ_py[i][2])", xlabel = "K", ylabel = "r")
    CairoMakie.heatmap!(ax, vals_heatmap_py[:, :, i], colorrange = (0, 1))
end
fig

ax5 = Axis(fig[4, 1], ylabel = "r", ylabelrotation = 0, rightspinevisible = false, topspinevisible = false,
    title = "Betabinomial(α = 1.0, β = K)")
CairoMakie.heatmap!(ax5, vals_heatmap_bb, colorrange = (0, clim_max))

ax6 = Axis(fig[4, 2], ylabel = "r", rightspinevisible = false, topspinevisible = false,
    title = "Dirichlet(α = H(K)⁻¹)")
CairoMakie.heatmap!(ax6, vals_heatmap_dp, colorrange = (0, clim_max))

fig
save("heatmap_py.pdf", fig)

#=
    TODO: the stuff below would be nice
using Optim

function objective_function(params)
    α, β, θ = params
    # Calculate the difference between the distributions
    difference = BetaBinomialPartitionDistribution(α, β) - PitmanYorProcess(θ)
    return difference
end

# Initial guess for the parameters
initial_params = [1.0, 1.0, 0.5]

# Define the optimization problem
problem = OptimizationProblem(objective_function, initial_params)

# Find the minimum using the Nelder-Mead algorithm
result = optimize(problem, NelderMead())

# Get the optimized parameters
optimized_params = Optim.minimizer(result)
=#

# import Interpolations
# function Interp2D(data, factor)

#     IC = Interpolations.CubicSplineInterpolation((axes(data,1), axes(data,2)), data)

#     finerx = LinRange(firstindex(data,1), lastindex(data,1), size(data,1) * factor)
#     finery = LinRange(firstindex(data,2), lastindex(data,2), size(data,2) * factor)
#     nx = length(finerx)
#     ny = length(finery)

#     data_interp = Array{Float64}(undef,nx,ny)
#     for i ∈ 1:nx, j ∈ 1:ny
#         if isnan(finerx[i]) || isnan(finery[j])
#             data_interp[i,j] = NaN
#         else
#             data_interp[i,j] = IC(finerx[i],finery[j])
#         end
#     end

#     return finery, finerx, data_interp
# end

vals_heatmap_bb = copy(vals_heatmap)
vals_heatmap_dp = permutedims(vals_heatmap)
for i in axes(vals_heatmap_bb, 1), j in i:size(vals_heatmap, 2)
    vals_heatmap_bb[i, j] = NaN
    vals_heatmap_dp[i, j] = NaN
end

m_bb = maximum(Iterators.filter(!isnan, vals_heatmap_bb))
m_dp = maximum(Iterators.filter(!isnan, vals_heatmap_dp))
clim_max = max(m_bb, m_dp)
no_ticks = 3
clim_ticks = range(0, clim_max, length = no_ticks)
# p3 = plot(heatmap(vals_heatmap_bb, clims = (0, clim_max)), xlab = "r", ylab = "K", title = "Betabinomial(α = 1.0, β = K)")
# p4 = plot(heatmap(vals_heatmap_dp, clims = (0, clim_max)), xlab = "r", ylab = "K", title = "Dirichlet(α = H(K)⁻¹)")
# p3 = plot(heatmap(vals_heatmap_bb, clims = (0, clim_max)), xlab = "r", ylab = "K", title = "Betabinomial(α = 1.0, β = K)")
# p4 = plot(heatmap(vals_heatmap_dp, clims = (0, clim_max)), xlab = "r", ylab = "K", title = "Dirichlet(α = H(K)⁻¹)")

ax5 = Axis(fig[1, 2], ylabel = "r", ylabelrotation = 0, rightspinevisible = false, topspinevisible = false,
    title = "Betabinomial(α = 1.0, β = K)")
CairoMakie.heatmap!(ax5, vals_heatmap_bb, colorrange = (0, clim_max))

ax6 = Axis(fig[1, 3], ylabel = "r", rightspinevisible = false, topspinevisible = false,
    title = "Dirichlet(α = H(K)⁻¹)")
CairoMakie.heatmap!(ax6, vals_heatmap_dp, colorrange = (0, clim_max))

Colorbar(fig[0, 3], limits = (0, clim_max), colormap = :viridis, size = 25, vertical = false)

fig




αβ_bb = (:uniform, :one_k, :binomial_k)#:uniform, )
α_dp  = (0.5, 1.0, 2.0, :harmonic, :GB)
# αθ_py = ((1.0, 0.0), (1.0, 0.2), (1.0, 0.5), (1.0, 0.9), (0.1, 1.0))
αθ_py = [(1.0, i) for i in 0.0:0.1:1.0]#, (1.0, 0.2), (1.0, 0.5), (1.0, 0.9), (0.1, 1.0))
max_k = 30

vals_heatmap_bb = fill(NaN, max_k, max_k, length(αβ_bb))
vals_heatmap_dp = fill(NaN, max_k, max_k, length(α_dp))
vals_heatmap_py = fill(NaN, max_k, max_k, length(αθ_py))


prog = ProgressMeter.Progress((max_k * (max_k - 1)) ÷ 2, showspeed=true)
for k in axes(vals_heatmap, 1)
    for r in 1:k
        if r != k
            for (j, method) in enumerate(αβ_bb)

                α, β = if method == :uniform
                    1., 1.
                elseif method == :one_k
                    1., float(k)
                elseif method == :binomial_k
                    1., max(1, binomial(k, 2))
                end
                if k > 30
                    vals_heatmap_bb[k, r, j] = Float64(equation_12(big(k), big(r), α, β))
                else
                    vals_heatmap_bb[k, r, j] = equation_12(k, r, α, β)
                end
            end
            for (j, method) in enumerate(α_dp)

                α = if method isa Number
                    method
                elseif method == :GB
                    EqualitySampler.dpp_find_α(k)
                elseif method == :harmonic
                    isone(k) ? 1.0 : 1 / harmonic_number(k - 1)
                end
                vals_heatmap_dp[k, r, j] = α / (α + k - 1)

            end
            for (j, method) in enumerate(αθ_py)
                θ, β = method
                vals_heatmap_py[k, r, j] = prediction_rule_PY(k, r, θ, β)
            end
        end
        ProgressMeter.next!(prog)
    end
end

fig = Figure(size = (4, 4) .* 400)
for i in axes(vals_heatmap_bb, 3)
    method = αβ_bb[i]
    α, β = if method == :uniform
        1, 1
    elseif method == :one_k
        1., "k"
    else
        1., "binomial(k, 2)"
    end
    ax = Axis(fig[i, 1], title = "BB: α = $α, β = $β", xlabel = "K", ylabel = "r")
    CairoMakie.heatmap!(ax, vals_heatmap_bb[:, :, i], colorrange = (0, 1))
end
for i in axes(vals_heatmap_dp, 3)
    method = α_dp[i]
    α = if method isa Number
        method
    elseif method == :GB
        "GB"
    elseif method == :harmonic
        "decreasing"
    end
    ax = Axis(fig[i, 2], title = "DP: α = $α", xlabel = "K", ylabel = "r")
    CairoMakie.heatmap!(ax, vals_heatmap_dp[:, :, i], colorrange = (0, 1))
end
for i in axes(vals_heatmap_py, 3)
    ax = Axis(fig[i, 3], title = "PY: α = $(αθ_py[i][1]), β = $(αθ_py[i][2])", xlabel = "K", ylabel = "r")
    CairoMakie.heatmap!(ax, vals_heatmap_py[:, :, i], colorrange = (0, 1))
end
fig


#region pretty interactive plot
import GLMakie
GLMakie.activate!(inline = false)
methods = (
    (
        name = "BB(α, β)",
        param_nms = ["α", "β"],
        param_ranges = [0:0.1:10, 0:0.1:10],
        param_starts = [1.0, 1.0],
        prediction_rule = (k, r, α, β) -> prediction_rule(BetaBinomialPartitionDistribution, r, k, α, β)
    ),
    (
        name = "BB(α, β*k)",
        param_nms = ["α", "β*k"],
        param_ranges = [0:0.1:10, 0:0.1:10],
        param_starts = [1.0, 1.0],
        prediction_rule = (k, r, α, β) -> prediction_rule(BetaBinomialPartitionDistribution, r, k, α, k * β)
    ),
    (
        name = "DP(α)",
        param_nms = ["α"],
        param_ranges = [0:0.1:10],
        param_starts = [1.0],
        prediction_rule = (k, r, α) -> prediction_rule(DirichletProcessPartitionDistribution, r, k, α)
    ),
    # (
    #     name = "DP(HN(k)⁻¹)",
    #     param_nms = String[],
    #     prediction_rule = (k, r) -> begin
    #         α = isone(k) ? 1.0 : 1 / harmonic_number(k - 1)
    #         α / (α + k - 1)
    #     end
    # ),
    (
        name = "Pitman Yor(θ, β)",
        param_nms = ["θ", "d"],
        param_ranges = [-1:0.1:10, 0:0.01:1],
        param_starts = [.5, .5],
        prediction_rule = function(k, r, θ, d)
            θ > -d || return NaN
            prediction_rule(PitmanYorProcessPartitionDistribution, r, k, d, θ)
        end
    )
)

max_k = 30
fig = Figure()
for i in eachindex(methods)
    method = methods[i]
    ax = Axis(fig[1, i], title = method.name, xlabel = "K", ylabel = "r")
    if length(method.param_nms) > 0
        labels = map(zip(method.param_nms, method.param_ranges, method.param_starts)) do (nm, range, startvalue)
            (label = nm, range = range, startvalue = startvalue)
        end

        sg = GLMakie.SliderGrid(fig[2, i], labels...)
        sliderobservables = [s.value for s in sg.sliders]
        vals_heatmap = lift(sliderobservables...) do slvalues...
            # α, β = slvalues
            @show methods[i].name, slvalues
            vals_heatmap = Matrix{Float64}(undef, max_k, max_k)
            fill!(vals_heatmap, NaN)
            for k in axes(vals_heatmap, 1), r in 1:k-1#:size(vals_heatmap, 2)
                # @show k, r, slvalues, method.name, methods[i].name, i
                vals_heatmap[k, r] = methods[i].prediction_rule(k, r, slvalues...)
            end
            vals_heatmap
        end
        GLMakie.heatmap!(ax, vals_heatmap, colorrange = (0, 1))
    else

        vals_heatmap = Matrix{Float64}(undef, max_k, max_k)
        fill!(vals_heatmap, NaN)
        for k in axes(vals_heatmap, 1), r in 1:k-1
            vals_heatmap[k, r] = method.prediction_rule(k, r)
        end
        GLMakie.heatmap!(ax, vals_heatmap, colorrange = (0, 1))
    end
end
GLMakie.Colorbar(fig[1, end + 1], limits = (0, 1), colormap = :viridis)
fig
#endregion


pdf(BetaBinomialPartitionDistribution(5, 1, 1), ones(Int, 5))
pdf(BetaBinomialPartitionDistribution(5, 1, 5), ones(Int, 5))
pdf(BetaBinomialPartitionDistribution(5, 1, 10), ones(Int, 5))


fig = Figure()
sg_bb = SliderGrid(fig[2, 1],
    (label = "α", range = 0:0.1:10, startvalue = 1),
    (label = "β", range = 0:0.1:10, startvalue = 1),
)
sg_bb_k = SliderGrid(fig[2, 2],
    (label = "α",   range = 0:0.1:10, startvalue = 1),
    (label = "β*k", range = 0:0.1:10, startvalue = 1),
)
sg_py = SliderGrid(fig[2, 3],
    (label = "α", range = 0:0.1:10, startvalue = 1),
    (label = "β", range = 0:0.1:10, startvalue = 1),
)


sliderobservables_bb = [s.value for s in sg_bb.sliders]
vals_heatmap_bb = lift(sliderobservables_bb...) do slvalues...
    α, β = slvalues
    vals_heatmap = Matrix{Float64}(undef, max_k, max_k)
    fill!(vals_heatmap, NaN)
    for k in axes(vals_heatmap, 1), r in 1:k-1#:size(vals_heatmap, 2)
        vals_heatmap[k, r] = equation_12(k, r, α, β)
    end
    vals_heatmap
end

sliderobservables_bb_k = [s.value for s in sg_bb_k.sliders]
vals_heatmap_bbk = lift(sliderobservables_bb_k...) do slvalues...
    α, β = slvalues
    vals_heatmap = Matrix{Float64}(undef, max_k, max_k)
    fill!(vals_heatmap, NaN)
    for k in axes(vals_heatmap, 1), r in 1:k-1#:size(vals_heatmap, 2)
        vals_heatmap[k, r] = equation_12(k, r, α, β*k)
    end
    vals_heatmap
end

sliderobservables_py = [s.value for s in sg_py.sliders]
vals_heatmap_py = lift(sliderobservables_py...) do slvalues...
    θ, β = slvalues
    vals_heatmap = Matrix{Float64}(undef, max_k, max_k)
    fill!(vals_heatmap, NaN)
    for k in axes(vals_heatmap, 1), r in 1:k-1#:size(vals_heatmap, 2)
        vals_heatmap[k, r] = prediction_rule_PY(k, r, θ, β)
    end
    vals_heatmap
end

Label(fig[0, 1], "BB(α, β)",     tellwidth = false, tellheight = true)
Label(fig[0, 2], "BB(α, β*k)",   tellwidth = false, tellheight = true)
Label(fig[0, 3], "PY(α, θ)",     tellwidth = false, tellheight = true)
GLMakie.heatmap!(Axis(fig[1, 1]), vals_heatmap_bb,  colorrange = (0, 1))
GLMakie.heatmap!(Axis(fig[1, 2]), vals_heatmap_bbk, colorrange = (0, 1))
GLMakie.heatmap!(Axis(fig[1, 3]), vals_heatmap_py,  colorrange = (0, 1))
GLMakie.Colorbar(fig[1, 4], limits = (0, 1), colormap = :viridis)
fig


# plot(heatmap(vals_heatmap_bb), cticks = collect(clim_ticks))

# samps_dp = rand(DirichletProcessPartitionDistribution(5, 1.0), 100)
# mean(col -> col[1] == col[2], eachcol(samps_dp))
# EqualitySampler.Simulations.compute_post_prob_eq(samps_dp')

# samps_bb = rand(BetaBinomialPartitionDistribution(5, 1.0, binomial(5, 2)), 10000)
# mean(col -> col[1] == col[2], eachcol(samps_bb))
# EqualitySampler.Simulations.compute_post_prob_eq(samps_bb')


# sample_one_equals_two(d)
# p4 = plot(heatmap(vals_heatmap #=Interp2D(vals_heatmap, 4)=#, clims = (0, 1)), xlab = "r\nDirichlet α = 1.0", ylab = "Betabinomial(α = 1.0, β = 1.0)\nK",
#     title = "Probability of including a new cluster")

# p_4panel = plot(p1, p2, p3, p4, size = (1000, 1000))
# savefig(p_4panel, "new_comparison_plot_4panel.pdf")

# p5 = plot(repeat(ks, length(models)), exp.(results_df.prob_null), group = results_df.model, legend = false,
#     title = "Probability of the null model", ylim = (0, 1), xlab = "K", linestyle = linestyle, linewidth = linesize, linecolor = linecolors)

# p6 = plot(repeat(ks, length(models)), results_df.prob_one_equals_two, group = results_df.model, legend = false,
#     title = "Probability θ₁ = θ₂", ylim = (0, 1), xlab = "K", linestyle = linestyle, linewidth = linesize, linecolor = linecolors)


ax3 = Axis(fig[2, 2], xticks = xticks, yticks = yticks, limits = lims, rightspinevisible = false, topspinevisible = false,
    title = "Probability of the null model")
# (i, df) = first(enumerate(results_df_grouped))
for (i, df) in enumerate(results_df_grouped)
    model = df.model[1]
    lines!(ax3, ks, exp.(df.prob_null), label = model, linestyle = linestyle_dict[model], linecolor = all_linecolors_dict[model], linewidth = 2)
end

ax4 = Axis(fig[2, 3], xticks = xticks, yticks = yticks, limits = lims, rightspinevisible = false, topspinevisible = false,
    title = "Probability θ₁ = θ₂")
# (i, df) = first(enumerate(results_df_grouped))
for (i, df) in enumerate(results_df_grouped)
    model = df.model[1]
    lines!(ax4, ks, df.prob_one_equals_two, label = model, linestyle = linestyle_dict[model], linecolor = all_linecolors_dict[model], linewidth = 2)
end

Label(fig[3, 1:3], "K")
fig

save("new_comparison_plot_6panel_fewer_priors_makie.pdf", fig)


import Turing
rand(RandomProcessPartitionDistribution(5, Turing.RandomMeasures.PitmanYorProcess(0.1, 0.1, 1)), 100)



# p_6panel = plot(p1, p2, p5, p6, p3, p4, layout = (3, 2), size = (1000, 1000))
p1256 = plot(plot(p1, p2), plot(p5, p6, bottom_margin = 1Plots.PlotMeasures.cm), layout = (2, 1), size = (1000, 1000))
p34 = plot(p3, p4, size = (1000, 500), plot_title = "Probability of a new cluster given the current amount of clusters")
p_6panel = plot(p1256, p34, layout = (2, 1), size = 750 .* (3, 2))

# savefig(p_6panel, "new_comparison_plot_6panel.pdf")
savefig(p_6panel, "new_comparison_plot_6panel_fewer_priors.pdf")



# random stuff starts below
p2b = plot(repeat(ks, length(models)), results_df.prob_big_normalized, group = results_df.model, legend = false,
    title = "Probability of joining the largest cluster", ylim = (0, 1), xlab = "K", linestyle = linestyle)



k, α, β = 5, 1, 1
d_bb = BetaBinomialPartitionDistribution(k, α, β)
d_dp = DirichletProcessPartitionDistribution(k, 1.0)

partition = [1, 1, 2, -1, -1]
θ_joint_bb = rand_θ_joint(10_000, d_bb, partition)
θ_joint_dp = rand_θ_joint(10_000, d_dp, partition)

# scatter(θ_joint[:, 1], θ_joint[:, 2])

default(fillcolor = :lightgrey)

markersize = .01
markeralpha = .2
markerstrokealpha = 0.0
x, y = θ_joint_bb[:, 1], θ_joint_bb[:, 2]

layout = @layout [a            b
                  b{0.8w,0.8h} c]
p3 = plot(histogram(x, xlim = (-5, 5), showaxis = false),                     plot(framestyle = :none, title = "BetaBinomial"),
     scatter(x,y, markersize = markersize, markeralpha = markeralpha, markerstrokealpha = markerstrokealpha, xlim = (-5, 5), ylim = (-5, 5), xlab = "θⱼ", ylab = "θⱼ₊₁"),     histogram(y, orientation = :horizontal, ylim = (-5, 5), showaxis = false),
     link = :both, layout = layout, legend = false)

x, y = θ_joint_dp[:, 1], θ_joint_dp[:, 2]

p4 = plot(histogram(x, xlim = (-5, 5), showaxis = false),                     plot(framestyle = :none, title = "Dirichlet"),
scatter(x,y, markersize = markersize, markeralpha = markeralpha, markerstrokealpha = markerstrokealpha, xlim = (-5, 5), ylim = (-5, 5), xlab = "θⱼ", ylab = "θⱼ₊₁"),     histogram(y, orientation = :horizontal, ylim = (-5, 5), showaxis = false),
link = :both, layout = layout, legend = false)


layout2 = @layout [a b
                   c d]

p_joined = plot(p1, p2, p3, p4, layout = layout2, size=(2, 2) .* 750)#, bottom_margin = 10Plots.PlotMeasures.mm)
savefig(p_joined, "new_comparison_plot.pdf")


default(fillcolor = :lightgrey, markercolor = :white, grid = false, legend = false)
plot(layout = layout, link = :both, size = (500, 500), margin = -10Plots.px)
scatter!(x,y, subplot = 2, framestyle = :box)
histogram!([x y], subplot = [1 3], orientation = [:v :h], framestyle = :none)


x, y = rand(Normal(), 3000), rand(TDist(2), 3000)

plot(    histogram(x),     plot(framestyle = :none),
         scatter(x,y),     histogram(y, orientation = :horizontal),
     link = :both)






α_dp = (.5, 1.0, 2.0, :Gopalan_Berry, :decreasing)
αβ_bb = (:uniform, :one_k, :one_binomial_k)

max_k = 30
vals_bb = fill(NaN, max_k, max_k, length(αβ_bb))
vals_dp = fill(NaN, max_k, max_k, length(α_dp))

prog = ProgressMeter.Progress((max_k * (max_k - 1)) ÷ 2, showspeed=true)
for k in axes(vals_bb, 1)
    for r in 1:k
        for (j, method) in enumerate(αβ_bb)

            α, β = if method == :uniform
                1., 1.
            elseif method == :one_k
                1., float(k)
            else
                1., max(1, binomial(k, 2))
            end
            if k > 30
                vals_bb[k, r, j] = Float64(equation_12(big(k), big(r), α, β))
            else
                vals_bb[k, r, j] = equation_12(k, r, α, β)
            end
        end
        for (j, method) in enumerate(α_dp)

            α = if j <= 3
                method
            elseif j == 4
                EqualitySampler.dpp_find_α(k)
            else
                isone(k) ? 1.0 : 1 / harmonic_number(k - 1)
            end
            vals_dp[k, r, j] = α / (α + k - 1)

        end
        ProgressMeter.next!(prog)
    end
end

# 2.0 ./ (2.0 .+ (1:50) .- 1)
# vals_dp[:, 10, :]
# something is wrong with dpp
# heatmap(vals_dp[:, :, 3])
# not wrong, just converging to 0 more slowly.

#=
k, r = 40, 34
equation_12(k,      r,      α, β)
equation_12(Int128(k),      Int128(r),      α, β)
equation_12(big(k), big(r), α, β)

extrema(Iterators.filter(!isnan, vals_bb))
extrema(Iterators.filter(!isnan, vals_dp))
=#

clims = (0.0, 1.0)

plts_bb = [
    plot(heatmap(vals_bb[:, :, i], clims = clims, title = Printf.@sprintf("Betabinomial: %s", string(αβ_bb[i]))))
    for i in axes(vals_bb, 3)
]

plts_dp = [
    plot(heatmap(vals_dp[:, :, i], clims = clims, title = Printf.@sprintf("Dirichlet: %s", string(α_dp[i]))))
    for i in axes(vals_dp, 3)
]

layout = @layout [
    a d
    b e
    c f
    g h
]

plts = Vector{Plots.Plot{Plots.GRBackend}}(undef, 8)
i_bb = i_dp = 1
for i in eachindex(plts)
    if isodd(i) && i <= 5
        plts[i] = plts_bb[i_bb]
        i_bb += 1
    else
        plts[i] = plts_dp[i_dp]
        i_dp += 1
    end
end

log_vals_bb = log.(vals_bb)
log_vals_dp = log.(vals_dp)

# ex_log_bb = extrema(Iterators.filter(x->!isnan(x) && isfinite(x), log_vals_bb))
# ex_log_dp = extrema(Iterators.filter(x->!isnan(x) && isfinite(x), log_vals_dp))
# clims_log = min(ex_log_bb[1], ex_log_dp[1]), max(ex_log_bb[2], ex_log_dp[2])

clims_log = (-50.0, 0.0)

plts_log_bb = [
    plot(heatmap(log_vals_bb[:, :, i], clims = clims_log, title = Printf.@sprintf("Betabinomial: %s", string(αβ_bb[i]))))
    for i in axes(vals_bb, 3)
]

plts_log_dp = [
    plot(heatmap(log_vals_dp[:, :, i], clims = clims_log, title = Printf.@sprintf("Dirichlet: %s", string(α_dp[i]))))
    for i in axes(vals_dp, 3)
]

plts_log = Vector{Plots.Plot{Plots.GRBackend}}(undef, 8)
i_bb = i_dp = 1
for i in eachindex(plts)
    if isodd(i) && i <= 5
        plts_log[i] = plts_log_bb[i_bb]
        i_bb += 1
    else
        plts_log[i] = plts_log_dp[i_dp]
        i_dp += 1
    end
end


default()
plt = plot(
    plts...,
    layout = layout,
    linkaxes = true,
    size=(2, 4) .* 750
)
plt_log = plot(
    plts_log...,
    layout = layout,
    linkaxes = true,
    size=(2, 4) .* 750
)
savefig(plt, "heatmap_bb_vs_dp.pdf")
savefig(plt_log, "heatmap_bb_vs_dp_logscale.pdf")


default()
plt = plot(
    heatmap(vals_bb, title = "Betabinomial"),
    heatmap(vals_dp, title = "Dirichlet"),
    linkaxes = true
)
plt_log = plot(
    heatmap(log.(vals_bb), title = "Betabinomial"),
    heatmap(log.(vals_dp), title = "Dirichlet"),
    linkaxes = true
)

savefig(plt, "heatmap_bb_vs_dp.pdf")


[
    exp.(logpdf_model_distinct(DirichletProcessPartitionDistribution(k, 1.0), ones(Int, k)))
    for k in 2:50
]


[
    exp.(logpdf_model_distinct(BetaBinomialPartitionDistribution(k, 1, k), ones(Int, k)))
    for k in 2:50
]
[
    exp.(logpdf_model_distinct(BetaBinomialPartitionDistribution(k, 1, 1), ones(Int, k)))
    for k in 2:50
]

[
    exp.(logpdf_model_distinct(BetaBinomialPartitionDistribution(k, 1, binomial(k, 2)), ones(Int, k)))
    for k in 2:50
]

[
    exp.(logpdf_model_distinct(BetaBinomialPartitionDistribution(k, 1, k-1), ones(Int, k)))
    for k in 2:50
]

[
    exp.(logpdf_model_distinct(BetaBinomialPartitionDistribution(k, 1, k), ones(Int, k)))
    for k in 2:50
]


ks = 8:20
[ks;; bellnum.(ks) ;; bellnum.(ks) .> 2^20]
2^20
12
4_213_597


[
    exp.(logpdf_model_distinct(DirichletProcessPartitionDistribution(k, 1.0), ones(Int, k)))
    for k in 2:50
]

k = 7
mm = collect(EqualitySampler.PartitionSpace(k))

log_probs_dp = [
    logpdf_model_distinct(DirichletProcessPartitionDistribution(k, .5), m)
    for m in mm
]

ms = EqualitySampler.no_distinct_groups_in_partition.(mm)
sortperm(ms)

new_ord = indexin(ms, 1:k)

grouped = [findall(==(i), ms) for i in 1:k]

log_probs_dp_grouped = [
    log(sum(exp, log_probs_dp[grouped[i]]))
    for i in eachindex(grouped)
]
exp.(log_probs_dp_grouped)
.5 * factorial(k-1) / prod(.5 + j - 1 for j in 1:k)
.5 ^k / prod(.5 + j - 1 for j in 1:k)

1/14

diff(log_probs_dp_grouped)


logpdf_model_distinct(DirichletProcessPartitionDistribution(k, .5), ones(Int, k))
logpdf_model_distinct(DirichletProcessPartitionDistribution(k, .5), 1:k)
logpdf_incl.(Ref(DirichletProcessPartitionDistribution(k, .5)), 1:k)


diff(logpdf_incl.(Ref(DirichletProcessPartitionDistribution(k, .5)), 1:k))

diff(pdf_incl.(Ref(DirichletProcessPartitionDistribution(k, .5)), 1:k))
incl_probs = pdf_incl.(Ref(DirichletProcessPartitionDistribution(k, .5)), 1:k)
incl_probs_ratio = incl_probs[1:end - 1] ./ incl_probs[2:end]
incl_probs_ratio .>= 1.
incl_probs_ratio[1]
H(k - 1)

round.(incl_probs; digits =3)

round.(incl_probs_ratio; digits =3)
@. (1 / 0.5) * exp(logunsignedstirlings1(k, 1:k-1) - logunsignedstirlings1(k, 2:k))

(1 / 0.5) * exp(logunsignedstirlings1(k, 1) - logunsignedstirlings1(k, 2))
factorial(k - 1)
factorial(k - 1) * H(k - 1)

1 / (H(k - 1) * .5)

H(k - 1)


incl_probs = pdf_incl.(Ref(DirichletProcessPartitionDistribution(k, 1/H(k - 1))), 1:k)
incl_probs_ratio = incl_probs[1:end - 1] ./ incl_probs[2:end]
incl_probs_ratio .>= 1.

.8 / .6
aa = 1e6
(.8 + aa) / (.6 + aa)

k, s = 20, 12
e1 = exp(logunsignedstirlings1(k, s))
e2 = exp(logunsignedstirlings1(k, s+1))
e1 / e2
others = [exp(logunsignedstirlings1(k, i)) for i in 1:k if i != s]
(e1 + sum(others)) / (e2 + sum(others))
factorial(big(k)) ≈ (e1 + sum(others))
factorial(big(k)) - e1 + e2 ≈ (e2 + sum(others))

e1 / e2
(e1 + sum(others)) / (e2 + sum(others))

1 / (1 + (e1 + e2))

sum(others) ≈ factorial(big(k)) - e1
(e1 + sum(others)) ≈ factorial(big(k)) - e1

e1 - factorial(big(k)) - e1




@. exp(logunsignedstirlings1(k, 0:k-1) - logunsignedstirlings1(k, 2:k))
exp.(logunsignedstirlings1.(k, 1:k-1) - logunsignedstirlings1.(k, 2:k))

all(<=(0), diff(log_probs_dp))


using EqualitySampler

precision(BigFloat)
setprecision(2^9)

nmax = 100
table_s1 = [
    # exp(logunsignedstirlings1(n, k))
    logunsignedstirlings1(n, k)
    for n in big(1):nmax, k in big(1):nmax
]

rows_s1_diff = [diff(@view table_s1[i, 1:i]) for i in axes(table_s1, 1)]
map(rows_s1_diff[2:end]) do row
    findmax(row)[2]
end
table_s1

rows_s1_diff

exp.(table_s1)

logunsignedstirlings1(10, 3)
prod(10 + i for i in 1:3)


n = 7
exp.(table_s1[n+1, 1:n+1])
n .* exp.(table_s1[n, 1:n+1]) .+ [1; exp.(table_s1[n, 1:n])]

import SpecialFunctions
col_1 = SpecialFunctions.logfactorial.(0:nmax - 1)
col_2 = (3 .* (1:nmax) .- 1) ./ 4 .+ SpecialFunctions.logabsbinomial.(0:nmax - 1)

table_s1[:, 1]

#1
exp.(logunsignedstirlings1.(1:10, 2))
2 .^ (1:10) .- 1

exp.(logunsignedstirlings1.(1:10, 1))
factorial.(0:9)

H(n) = sum(1 / big(i) for i in n:-1:1)

exp.(logunsignedstirlings1.(1:10, 2))
factorial.(1:9) .* H.(1:9)

kk = 10
[Float64(exp(logunsignedstirlings1(big(kk), i) - logunsignedstirlings1(big(kk), i+1))) for i in 1:kk-1]
kk = 20
[Float64(exp(logunsignedstirlings1(big(kk), i) - logunsignedstirlings1(big(kk), i+1))) for i in 1:kk-1]



+ factorial.(0:10 - 1)

exp(table_s1[5, 3])
4 * exp(table_s1[4, 3]) + exp(table_s1[4, 2])

exp(table_s1[4, 3])
(exp(table_s1[5, 3]) - exp(table_s1[4, 2])) / 4

exp(table_s1[4, 2])
(exp(table_s1[5, 2]) - exp(table_s1[4, 1])) / 4

exp(table_s1[4, 3]) / exp(table_s1[4, 2])
(exp(table_s1[5, 3]) - exp(table_s1[4, 2])) / (exp(table_s1[5, 2]) - exp(table_s1[4, 1]))

#=
    conclusion:

    difference is maximal for k = 1 to k = 2
    s(n, 1) =

=#
# difference in



nmax = 500
table_s1 = [
    # exp(logunsignedstirlings1(n, k))
    logunsignedstirlings1(n, k)
    for n in big(1):nmax, k in big(1):3
]

all(table_s1[i, 1] - table_s1[i, 2] < table_s1[i, 2] - table_s1[i, 3]
    for i in 2:nmax)