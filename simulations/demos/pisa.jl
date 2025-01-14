using EqualitySampler
import JLD2, CodecZlib, CSV, DataFrames as DF, SpecialFunctions, Printf
import AlgebraOfGraphics as AOG
import Distributions
import CairoMakie as CM
import StatsBase as SB, LinearAlgebra as LA, NamedArrays as NA
import Chain: @chain
import ColorSchemes, Colors, OrderedCollections
import GraphMakie, NetworkLayout, Graphs # TODO: remove?

include("../utilities.jl")

round_2_decimals(x::Number) = Printf.@sprintf "%.2f" x
round_2_decimals(x) = x

function run_analyses(results_dir, data_file, force)

    pisa_data = CSV.read(data_file, DF.DataFrame)
    pisa_suffstats = EqualitySampler.extract_suffstats_one_way_anova(pisa_data.point_estimate, pisa_data.sd .^ 2, pisa_data.sample_size)

    k = length(pisa_suffstats.n_by_group)
    prior0 = BetaBinomialPartitionDistribution(k, 1.0, k)
    prior  = PrecomputedCustomInclusionPartitionDistribution(prior0)

    filename_eq = joinpath(results_dir, "pisa_eq.jld2")
    if !isfile(filename_eq) || force
        log_message("Starting equality analysis for Pisa data")

        initial_partition, _, _, _ = EqualitySampler.find_initial_partition(pisa_data.lower_ci, pisa_data.upper_ci, no_repeats = 10)
        method  = EqualitySampler.SampleRJMCMC(iter = 20_000, initial_partition = Int8.(initial_partition), max_cache_size = 100_000_000, split_merge_prob = 0.5)
        samples_eq = anova_test(pisa_suffstats, method, prior, threaded = true)
        JLD2.jldsave(filename_eq, true; samples = samples_eq)
    else
        log_message("Loading equality analysis for Pisa data from disk")
        samples_eq = JLD2.load(filename_eq, "samples")
    end

    filename_full = joinpath(results_dir, "pisa_full.jld2")
    if !isfile(filename_full) || force
        log_message("Starting full analysis for Pisa data from disk")
        samples_full = anova_test(pisa_suffstats, EqualitySampler.SampleRJMCMC(iter = 20_000, fullmodel_only = true), prior)
        JLD2.jldsave(filename_full, true; samples = samples_full)
    else
        log_message("Loading full analysis for Pisa data from disk")
        samples_full = JLD2.load(filename_full, "samples")
    end

    eq_probs = compute_post_prob_eq(samples_eq)
    threshold = 1 - tie_probability(prior0)
    mpm_partition = get_mpm_partition(eq_probs, threshold)

    filename_mpm = joinpath(results_dir, "pisa_mpm.jld2")
    if !isfile(filename_mpm)
        log_message("Starting mpm analysis for Pisa data from disk")
        samples_mpm = anova_test(pisa_suffstats, EqualitySampler.SampleRJMCMC(iter = 20_000, fullmodel_only = true, initial_partition = mpm_partition), prior0)
        JLD2.jldsave(filename_mpm, true; samples = samples_mpm, mpm_partition = mpm_partition)
    else
        log_message("Loading mpm analysis for Pisa data from disk")
        samples_mpm, mpm_partition = JLD2.load(filename_mpm, "samples", "mpm_partition")
    end

    return (; pisa_data, samples_eq, samples_full, samples_mpm, mpm_partition)
end

function produce_figures(figures_dir, results_obj)

    log_message("Creating figures for pisa analysis")
    (; samples_full, samples_eq, samples_mpm, mpm_partition) = results_obj

    cell_samples_full = samples_full.parameter_samples.θ_cp .+ samples_full.parameter_samples.μ'
    cell_samples_eq   = samples_eq.parameter_samples.θ_cp   .+ samples_eq.parameter_samples.μ'
    cell_samples_mpm  = samples_mpm.parameter_samples.θ_cp  .+ samples_mpm.parameter_samples.μ'

    cri_level = .95
    Δ_cri = (1 - cri_level) / 2

    pisa_data.est_full           = vec(SB.mean(cell_samples_full, dims = 2))
    pisa_data.est_full_lower_cri = SB.quantile.(eachrow(cell_samples_full), Δ_cri)
    pisa_data.est_full_upper_cri = SB.quantile.(eachrow(cell_samples_full), 1 - Δ_cri)

    pisa_data.est_eq           = vec(SB.mean(cell_samples_eq, dims = 2))
    pisa_data.est_eq_lower_cri = SB.quantile.(eachrow(cell_samples_eq), Δ_cri)
    pisa_data.est_eq_upper_cri = SB.quantile.(eachrow(cell_samples_eq), 1 - Δ_cri)

    pisa_data.est_mpm           = vec(SB.mean(cell_samples_mpm, dims = 2))
    pisa_data.est_mpm_lower_cri = SB.quantile.(eachrow(cell_samples_mpm), Δ_cri)
    pisa_data.est_mpm_upper_cri = SB.quantile.(eachrow(cell_samples_mpm), 1 - Δ_cri)

    one2one = AOG.mapping([0], [1]) * AOG.visual(CM.ABLines, color=:gray, linestyle=:dash)
    top = one2one + AOG.data(pisa_data) * (
        AOG.mapping(:point_estimate, :est_full, color = :subregion) * (
            (AOG.visual(CM.Errorbars; direction = :y, alpha = .25) * AOG.mapping((:est_full_lower_cri, :est_full_upper_cri) => (x, y) -> abs(x - y))) +
            AOG.visual(CM.Scatter; alpha = .75))
    )
    middle = one2one + AOG.data(pisa_data) * (
        AOG.mapping(:point_estimate, :est_eq, color = :subregion) * (
            (AOG.visual(CM.Errorbars; direction = :y, alpha = .25) * AOG.mapping((:est_eq_lower_cri, :est_eq_upper_cri) => (x, y) -> abs(x - y))) +
            AOG.visual(CM.Scatter; alpha = .75))
    )
    bottom = one2one + AOG.data(pisa_data) * (
        AOG.mapping(:point_estimate, :est_mpm, color = :subregion) * (
            (AOG.visual(CM.Errorbars; direction = :y, alpha = .25) * AOG.mapping((:est_mpm_lower_cri, :est_mpm_upper_cri) => (x, y) -> abs(x - y))) +
            AOG.visual(CM.Scatter; alpha = .75))
    )

    fig = CM.Figure();
    ax_top    = CM.Axis(fig[1, 1], title = "Full model",               titlefont = :regular)
    ax_middle = CM.Axis(fig[2, 1], title = "Model Averaged",           titlefont = :regular, ylabel = "Posterior mean + 95% CRI")
    ax_bottom = CM.Axis(fig[3, 1], title = "Median Probability Model", titlefont = :regular, xlabel = "Observed + 95% CI")
    CM.linkaxes!(ax_top, ax_middle)
    CM.linkaxes!(ax_top, ax_bottom)
    AOG.draw!(ax_top, top)
    AOG.draw!(ax_middle, middle)
    AOG.draw!(ax_bottom, bottom)
    CM.resize!(fig, 1200, 800)
    CM.save(joinpath(figures_dir, "pisa_full_vs_eq_vs_mpm.pdf"), fig)


    u_mpm_partition = unique(mpm_partition)
    summary_df = DF.DataFrame(
        countries      = Vector{Vector{String}}(undef, length(u_mpm_partition)),
        posterior_mean = Vector{Float64}(       undef, length(u_mpm_partition)),
        lower_cri      = Vector{Float64}(       undef, length(u_mpm_partition)),
        upper_cri      = Vector{Float64}(       undef, length(u_mpm_partition)),

        point_ests     = Vector{Vector{Float64}}(       undef, length(u_mpm_partition))
    )

    for (i, u) in enumerate(u_mpm_partition)
        idx = findall(==(u), mpm_partition)
        idx2 = sortperm(pisa_data.point_estimate[idx], rev = true)
        # idx2 = sortperm(pisa_data.country[idx], rev = true)
        summary_df.countries[i] = pisa_data.country[idx][idx2]
        cell_samples_v = view(cell_samples_mpm, idx[1], :)
        summary_df.posterior_mean[i] = SB.mean(cell_samples_v)
        summary_df.lower_cri[i] = SB.quantile(cell_samples_v, Δ_cri)
        summary_df.upper_cri[i] = SB.quantile(cell_samples_v, 1 - Δ_cri)

        summary_df.point_ests[i] = pisa_data.point_estimate[idx][idx2]
    end
    # length.(summary_df.countries)
    # maximum(length, summary_df.countries)

    # fig, ax, _ = CM.scatter(summary_df.posterior_mean, SB.mean.(summary_df.point_ests))
    # CM.ablines!(ax, 0, 1, color = :gray, linestyle = :dash)
    # fig

    sort!(summary_df, [:posterior_mean, :lower_cri, :upper_cri], rev = true)

    # mpm_partition_countries = mapreduce(sort, vcat, summary_df.countries)
    mpm_partition_countries = reduce(vcat, summary_df.countries)
    for i in eachindex(summary_df.countries)
        mpm_partition_countries[i] = replace(mpm_partition_countries[i], " " => "_")
    end

    yticks  = 300:50:600
    ylimits = collect(extrema(yticks))
    xticks  = [1:5:25;26]
    xlimits = [0.5, 28.0]#collect(extrema(xticks))
    colors = CM.Makie.wong_colors()[1:2]

    fig = CM.Figure(fontsize = 15)
    ax = CM.Axis(fig[1, 1], title = "Median Posterior Model", xlabel = "Rank", ylabel = "PISA score",
        xticks = xticks, yticks = yticks, limits = (xlimits[1], xlimits[2], ylimits[1], ylimits[2]))
    CM.scatter!(ax, axes(summary_df, 1), summary_df.posterior_mean)
    CM.errorbars!(ax, axes(summary_df, 1), summary_df.posterior_mean, summary_df.posterior_mean - summary_df.lower_cri, summary_df.upper_cri - summary_df.posterior_mean)

    ttt=reduce(vcat, summary_df.point_ests)
    yyy=reduce(vcat, fill.(axes(summary_df, 1), length.(summary_df.point_ests)))
    CM.scatter!(ax, yyy .+ .2, ttt, color = (colors[2], .4))

    isdiv3(i) = iszero(i % 3)
    isdiv4(i) = iszero(i % 4)
    offsets = [
        fill((10, 15),  4)  ;
        [
            iseven(i) ? (5, isdiv4(i) ? 15 : 20) : (-5, isdiv3(i) ? -15 : -20)
            for i in 1:11
        ] ;
        fill((10, 15), 10)  ;
        (8, 15)
    ]

    aligns = [
        fill((:center, :bottom),  4) ;
        [
            iseven(i) ? (:center, :bottom) : (:center, :top)
            for i in 1:11
        ] ;
        fill((:center, :bottom),  10);
        (:center, :bottom)
    ]
    points = CM.Point2.(axes(summary_df, 1), summary_df.posterior_mean)
    istart = 1
    for i in eachindex(summary_df.countries)
        istop = istart + length(summary_df.countries[i]) - 1
        # CM.text!(ax, i, summary_df.posterior_mean[i], text = first(sort(summary_df.countries[i])), align = aligns[i],
            # offset = offsets[i])

        otherpoint = CM.Point2(i + offsets[i][1] / 10, summary_df.posterior_mean[i] + offsets[i][2])
        # CM.text!(ax, otherpoint, text = first(sort(summary_df.countries[i])), align = aligns[i])

        txt = istart == istop ? string(istart) : (string(istart) * "-" * string(istop))
        CM.text!(ax, otherpoint, text = txt, align = aligns[i])
        # CM.scatter!(ax, i + offsets[i][1] / 25, summary_df.posterior_mean[i] + offsets[i][2], color = :grey)
        CM.lines!(ax, [points[i], otherpoint], color = :grey, linestyle = :dash)

        istart = istop + 1

    end

    resize!(fig, 1200, 700)
    fig

    ucolors = [col for (col, _) in zip(Iterators.cycle(Colors.distinguishable_colors(8)), 1:26)]
    colors = ucolors[mpm_partition2]
    labels = [string(i, ". ", replace(mpm_partition_countries[i], "_" => " ")) for i in eachindex(mpm_partition_countries)]
    legend_elements = [CM.MarkerElement(marker = :circle, markersize = 0) for label in labels]

    legend = CM.Legend(
        fig[1, 1],
        # gl[1, 2],
        legend_elements[1:50],
        labels[1:50],
        # "Legend",
        # labelcolor = colors,
        nbanks = 10,
        tellwidth = false,
        tellheight = false,
        orientation = :horizontal,  # Horizontal layout for columns
        framevisible = false,
        margin = ntuple(_->5, 4),
        padding = (0.0f0, 0.0f0, 0.0f0, 0.0f0),
        patchlabelgap = 0,
        halign = :left,
        valign = :bottom,
        rowgap = 2
    )
    legend = CM.Legend(
        fig[1, 1],
        # gl[1, 2],
        legend_elements[51:end],
        labels[51:end],
        # "Legend",
        # labelcolor = colors,
        nbanks = 7,
        tellwidth = false,
        tellheight = false,
        orientation = :horizontal,  # Horizontal layout for columns
        framevisible = false,
        margin = ntuple(_->5, 4),
        padding = (0.0f0, 0.0f0, 0.0f0, 0.0f0),
        patchlabelgap = 0,
        halign = :right,
        valign = :top,
        rowgap = 2
    )
    fig

    CM.save(joinpath(figures_dir, "pisa_mpm_with_names.pdf"), fig, pt_per_unit = 1)

end

function main(; data_file::String, results_dir::String, figures_dir::String, force::Bool = false)

    !isfile(data_file) && error("Data file not found: $data_file")
    analysis_results = run_analyses(results_dir, figures_dir, force)
    produce_figures(figures_dir, analysis_results)
end

main(
    data_file   = "simulations/demos/data/pisa.csv",
    results_dir = joinpath(pwd(), "simulations", "saved_objects"),
    figures_dir = joinpath(pwd(), "simulations", "revision_figures")
)

#=

using EqualitySampler
import JLD2, CodecZlib
import CSV
import DataFrames as DF
import SpecialFunctions
import Printf


round_2_decimals(x::Number) = Printf.@sprintf "%.2f" x
round_2_decimals(x) = x

if isinteractive()
    import AlgebraOfGraphics as AOG
    import Distributions
    import CairoMakie as CM
    import	StatsBase 		as SB,
            LinearAlgebra	as LA,
            NamedArrays		as NA
    import Chain: @chain
    import ColorSchemes, Colors, OrderedCollections
    import GraphMakie, NetworkLayout, Graphs
end
include("../utilities.jl")




results_dir
figures_dir


pisa_data = CSV.read("simulations/demos/data/pisa.csv", DF.DataFrame)

pisa_suffstats = EqualitySampler.extract_suffstats_one_way_anova(pisa_data.point_estimate, pisa_data.sd .^ 2, pisa_data.sample_size)

k = length(pisa_suffstats.n_by_group)
prior0 = BetaBinomialPartitionDistribution(k, 1.0, k)
prior  = PrecomputedCustomInclusionPartitionDistribution(prior0)

filename_eq = "/home/don/hdd/postdoc/equalitysampler/pisa_eq.jld2"
if !isfile(filename_eq)

    initial_partition, _, _, _ = EqualitySampler.find_initial_partition(pisa_data.lower_ci, pisa_data.upper_ci, no_repeats = 10)


    method  = EqualitySampler.SampleRJMCMC(iter = 20_000, initial_partition = Int8.(initial_partition), max_cache_size = 100_000_000, split_merge_prob = 0.5)
    samples_eq = anova_test(pisa_suffstats, method, prior, threaded = true)

    JLD2.jldsave(filename_eq, true; samples = samples_eq)
else
    samples_eq = JLD2.load(filename_eq, "samples")
end

filename_full = "/home/don/hdd/postdoc/equalitysampler/pisa_full.jld2"
if !isfile(filename_full)

    samples_full = anova_test(pisa_suffstats,
        EqualitySampler.SampleRJMCMC(iter = 20_000, fullmodel_only = true),
        prior
    )
    JLD2.jldsave(filename_full, true; samples = samples_full)
else
    samples_full = JLD2.load(filename_full, "samples")
end

eq_probs      = compute_post_prob_eq(samples_eq)
threshold = 1 - tie_probability(prior0)
mpm_partition = get_mpm_partition(eq_probs, threshold)

EqualitySampler.compute_incl_counts(samples_eq)
length(unique(mpm_partition))

filename_mpm = "/home/don/hdd/postdoc/equalitysampler/pisa_mpm.jld2"
if !isfile(filename_mpm)

    samples_mpm = anova_test(pisa_suffstats,
        EqualitySampler.SampleRJMCMC(iter = 20_000, fullmodel_only = true, initial_partition = mpm_partition),
        prior0
    )
    JLD2.jldsave(filename_mpm, true; samples = samples_mpm)

else
    samples_mpm = JLD2.load(filename_mpm, "samples")
end

!isinteractive() && exit()

@assert !any(map(allunique, eachcol(samples_eq.parameter_samples.θ_cp)))
@assert all(map(allunique, eachcol(samples_full.parameter_samples.θ_cp)))

cell_samples_eq   = samples_eq.parameter_samples.θ_cp   .+ samples_eq.parameter_samples.μ'
cell_samples_full = samples_full.parameter_samples.θ_cp .+ samples_full.parameter_samples.μ'
cell_samples_mpm  = samples_mpm.parameter_samples.θ_cp  .+ samples_mpm.parameter_samples.μ'


cri_level = .95
Δ_cri = (1 - cri_level) / 2

pisa_data.est_full           = vec(SB.mean(cell_samples_full, dims = 2))
pisa_data.est_full_lower_cri = SB.quantile.(eachrow(cell_samples_full), Δ_cri)
pisa_data.est_full_upper_cri = SB.quantile.(eachrow(cell_samples_full), 1 - Δ_cri)

pisa_data.est_eq           = vec(SB.mean(cell_samples_eq, dims = 2))
pisa_data.est_eq_lower_cri = SB.quantile.(eachrow(cell_samples_eq), Δ_cri)
pisa_data.est_eq_upper_cri = SB.quantile.(eachrow(cell_samples_eq), 1 - Δ_cri)

pisa_data.est_mpm           = vec(SB.mean(cell_samples_mpm, dims = 2))
pisa_data.est_mpm_lower_cri = SB.quantile.(eachrow(cell_samples_mpm), Δ_cri)
pisa_data.est_mpm_upper_cri = SB.quantile.(eachrow(cell_samples_mpm), 1 - Δ_cri)


one2one = AOG.mapping([0], [1]) * AOG.visual(CM.ABLines, color=:gray, linestyle=:dash)
top = one2one + AOG.data(pisa_data) * (
    AOG.mapping(:point_estimate, :est_full, color = :subregion) * (
        # (AOG.visual(CM.Errorbars; direction = :x, alpha = .25) * AOG.mapping((:lower_ci, :upper_ci) => (x, y) -> abs(x - y))) +
        (AOG.visual(CM.Errorbars; direction = :y, alpha = .25) * AOG.mapping((:est_full_lower_cri, :est_full_upper_cri) => (x, y) -> abs(x - y))) +
        AOG.visual(CM.Scatter; alpha = .75))
)
bottom = one2one + AOG.data(pisa_data) * (
    AOG.mapping(:point_estimate, :est_eq, color = :subregion) * (
        # (AOG.visual(CM.Errorbars; direction = :x, alpha = .25) * AOG.mapping((:lower_ci, :upper_ci) => (x, y) -> abs(x - y))) +
        (AOG.visual(CM.Errorbars; direction = :y, alpha = .25) * AOG.mapping((:est_eq_lower_cri, :est_eq_upper_cri) => (x, y) -> abs(x - y))) +
        AOG.visual(CM.Scatter; alpha = .75))
)

fig = CM.Figure();
ax_top    = CM.Axis(fig[1, 1], title = "Full model",     titlefont = :regular, ylabel = "Posterior mean + 95% CRI")
ax_bottom = CM.Axis(fig[2, 1], title = "Model Averaged", titlefont = :regular, ylabel = "Posterior mean + 95% CRI", xlabel = "Observed + 95% CI")
CM.linkaxes!(ax_top, ax_bottom)
AOG.draw!(ax_top, top)
AOG.draw!(ax_bottom, bottom)
CM.resize!(fig, 1200, 800)
fig

pisa_data.est_mpm           = vec(SB.mean(cell_samples_mpm, dims = 2))
pisa_data.est_mpm_lower_cri = SB.quantile.(eachrow(cell_samples_eq), Δ_cri)
pisa_data.est_mpm_upper_cri = SB.quantile.(eachrow(cell_samples_eq), 1 - Δ_cri)


one2one = AOG.mapping([0], [1]) * AOG.visual(CM.ABLines, color=:gray, linestyle=:dash)
top = one2one + AOG.data(pisa_data) * (
    AOG.mapping(:point_estimate, :est_full, color = :subregion) * (
        # (AOG.visual(CM.Errorbars; direction = :x, alpha = .25) * AOG.mapping((:lower_ci, :upper_ci) => (x, y) -> abs(x - y))) +
        (AOG.visual(CM.Errorbars; direction = :y, alpha = .25) * AOG.mapping((:est_full_lower_cri, :est_full_upper_cri) => (x, y) -> abs(x - y))) +
        AOG.visual(CM.Scatter; alpha = .75))
)
middle = one2one + AOG.data(pisa_data) * (
    AOG.mapping(:point_estimate, :est_eq, color = :subregion) * (
        # (AOG.visual(CM.Errorbars; direction = :x, alpha = .25) * AOG.mapping((:lower_ci, :upper_ci) => (x, y) -> abs(x - y))) +
        (AOG.visual(CM.Errorbars; direction = :y, alpha = .25) * AOG.mapping((:est_eq_lower_cri, :est_eq_upper_cri) => (x, y) -> abs(x - y))) +
        AOG.visual(CM.Scatter; alpha = .75))
)
bottom = one2one + AOG.data(pisa_data) * (
    AOG.mapping(:point_estimate, :est_mpm, color = :subregion) * (
        # (AOG.visual(CM.Errorbars; direction = :x, alpha = .25) * AOG.mapping((:lower_ci, :upper_ci) => (x, y) -> abs(x - y))) +
        (AOG.visual(CM.Errorbars; direction = :y, alpha = .25) * AOG.mapping((:est_mpm_lower_cri, :est_mpm_upper_cri) => (x, y) -> abs(x - y))) +
        AOG.visual(CM.Scatter; alpha = .75))
)

fig = CM.Figure();
ax_top    = CM.Axis(fig[1, 1], title = "Full model",               titlefont = :regular)
ax_middle = CM.Axis(fig[2, 1], title = "Model Averaged",           titlefont = :regular, ylabel = "Posterior mean + 95% CRI")
ax_bottom = CM.Axis(fig[3, 1], title = "Median Probability Model", titlefont = :regular, xlabel = "Observed + 95% CI")
CM.linkaxes!(ax_top, ax_middle)
CM.linkaxes!(ax_top, ax_bottom)
AOG.draw!(ax_top, top)
AOG.draw!(ax_middle, middle)
AOG.draw!(ax_bottom, bottom)
CM.resize!(fig, 1200, 800)
fig


u_mpm_partition = unique(mpm_partition)
summary_df = DF.DataFrame(
    countries      = Vector{Vector{String}}(undef, length(u_mpm_partition)),
    posterior_mean = Vector{Float64}(       undef, length(u_mpm_partition)),
    lower_cri      = Vector{Float64}(       undef, length(u_mpm_partition)),
    upper_cri      = Vector{Float64}(       undef, length(u_mpm_partition)),

    point_ests     = Vector{Vector{Float64}}(       undef, length(u_mpm_partition))
)

for (i, u) in enumerate(u_mpm_partition)
    idx = findall(==(u), mpm_partition)
    summary_df.countries[i] = sort!(pisa_data.country[idx])
    cell_samples_v = view(cell_samples_mpm, idx[1], :)
    summary_df.posterior_mean[i] = SB.mean(cell_samples_v)
    summary_df.lower_cri[i] = SB.quantile(cell_samples_v, Δ_cri)
    summary_df.upper_cri[i] = SB.quantile(cell_samples_v, 1 - Δ_cri)

    summary_df.point_ests[i] = pisa_data.point_estimate[idx]
end
length.(summary_df.countries)
maximum(length, summary_df.countries)

fig, ax, _ = CM.scatter(summary_df.posterior_mean, SB.mean.(summary_df.point_ests))
CM.ablines!(ax, 0, 1, color = :gray, linestyle = :dash)
fig



yticks  = 300:50:600
ylimits = collect(extrema(yticks))
xticks  = [1:5:25;26]
xlimits = [0.5, 28.0]#collect(extrema(xticks))
colors = CM.Makie.wong_colors()[1:2]
fig = CM.Figure()
ax = CM.Axis(fig[1, 1], title = "Median Posterior Model", xlabel = "Rank", ylabel = "PISA score",
    xticks = xticks, yticks = yticks, limits = (xlimits[1], xlimits[2], ylimits[1], ylimits[2]))
CM.scatter!(ax, axes(summary_df, 1), summary_df.posterior_mean)
CM.errorbars!(ax, axes(summary_df, 1), summary_df.posterior_mean, summary_df.posterior_mean - summary_df.lower_cri, summary_df.upper_cri - summary_df.posterior_mean)

ttt=reduce(vcat, summary_df.point_ests)
yyy=reduce(vcat, fill.(axes(summary_df, 1), length.(summary_df.point_ests)))
CM.scatter!(ax, yyy .+ .2, ttt, color = (colors[2], .4))

isdiv3(i) = iszero(i % 3)
isdiv4(i) = iszero(i % 4)
offsets = [
    fill((10, 15),  4)  ;
    [
        iseven(i) ? (5, isdiv4(i) ? 15 : 20) : (-5, isdiv3(i) ? -15 : -20)
        for i in 1:11
    ] ;
    fill((10, 15), 10)  ;
    (8, 15)
]

aligns = [
    fill((:center, :bottom),  4) ;
    [
        iseven(i) ? (:center, :bottom) : (:center, :top)
        for i in 1:11
    ] ;
    fill((:center, :bottom),  10);
    (:center, :bottom)
]
points = CM.Point2.(axes(summary_df, 1), summary_df.posterior_mean)
istart = 1
for i in eachindex(summary_df.countries)
    istop = istart + length(summary_df.countries[i]) - 1
    # CM.text!(ax, i, summary_df.posterior_mean[i], text = first(sort(summary_df.countries[i])), align = aligns[i],
        # offset = offsets[i])

    otherpoint = CM.Point2(i + offsets[i][1] / 10, summary_df.posterior_mean[i] + offsets[i][2])
    # CM.text!(ax, otherpoint, text = first(sort(summary_df.countries[i])), align = aligns[i])

    txt = istart == istop ? string(istart) : (string(istart) * "-" * string(istop))
    CM.text!(ax, otherpoint, text = txt, align = aligns[i])
    # CM.scatter!(ax, i + offsets[i][1] / 25, summary_df.posterior_mean[i] + offsets[i][2], color = :grey)
    CM.lines!(ax, [points[i], otherpoint], color = :grey, linestyle = :dash)

    istart = istop + 1

end

resize!(fig, 1200, 700)
fig

# layout = Point2.(yyy, ttt)
layout = [
    CM.Point2(i + offsets[i][1] / 10, summary_df.posterior_mean[i] + offsets[i][2])
    for i in eachindex(summary_df.countries)
]

ucolors = [col for (col, _) in zip(Iterators.cycle(Colors.distinguishable_colors(8)), 1:26)]
colors = ucolors[mpm_partition2]
# labels = [CM.rich(string(i, ". ", replace(mpm_partition_countries[i], "_" => " ")), color = colors[i]) for i in eachindex(mpm_partition_countries)]
labels = [string(i, ". ", replace(mpm_partition_countries[i], "_" => " ")) for i in eachindex(mpm_partition_countries)]
legend_elements = [CM.MarkerElement(marker = :circle, markersize = 0) for label in labels]

legend = CM.Legend(
    fig[1, 1],
    # gl[1, 2],
    legend_elements[1:50],
    labels[1:50],
    # "Legend",
    # labelcolor = colors,
    nbanks = 10,
    tellwidth = false,
    tellheight = false,
    orientation = :horizontal,  # Horizontal layout for columns
    framevisible = false,
    margin = ntuple(_->5, 4),
    padding = (0.0f0, 0.0f0, 0.0f0, 0.0f0),
    patchlabelgap = 0,
    halign = :left,
    valign = :bottom,
    rowgap = 2
)
legend = CM.Legend(
    fig[1, 1],
    # gl[1, 2],
    legend_elements[51:end],
    labels[51:end],
    # "Legend",
    # labelcolor = colors,
    nbanks = 7,
    tellwidth = false,
    tellheight = false,
    orientation = :horizontal,  # Horizontal layout for columns
    framevisible = false,
    margin = ntuple(_->5, 4),
    padding = (0.0f0, 0.0f0, 0.0f0, 0.0f0),
    patchlabelgap = 0,
    halign = :right,
    valign = :top,
    rowgap = 2
)
fig


CM.save("simulations/newfigures/pisa_mpm_with_names_all_in_one.pdf", fig,
    pt_per_unit = 1)


# TODO: order this by rank so Singapore is 1, etc.
mpm_partition_countries = mapreduce(sort, vcat, summary_df.countries)
indexin(mpm_partition_countries, pisa_data.country)
mpm_partition2 = mpm_partition[indexin(mpm_partition_countries, pisa_data.country)]

mpm_adj = mpm_partition2 .== mpm_partition2'
mpm_adj[LinearAlgebra.diagind(mpm_adj)] .= false
mpm_graph = Graphs.SimpleGraph(mpm_adj)
# mpm_partition_countries .== mpm_partition_countries'

function get_xy(k::Integer)
	offset = k == 4 ? pi / 4 : 0.0
	x = Float64[sin(i * 2pi / k + offset) for i in 1:k]
	y = Float64[cos(i * 2pi / k + offset) for i in 1:k]
	return x, y
end

size(summary_df, 1)

swap_last_to_first(x) = x[[lastindex(x); firstindex(x):(lastindex(x) - 1)]]
ringsizes = [15, 11]
rings = [
    swap_last_to_first(CM.Point2.(get_xy(ringsize)...))
    for ringsize in ringsizes
]
# ring_idx = [
#     (1:ringsize) .+ sum(view(ringsizes, 1:i-1); init = 0)
#     for (i, ringsize) in enumerate(ringsizes)
# ]
ring_idx = reduce(vcat, [
    fill(i, ringsize)
    for (i, ringsize) in enumerate(ringsizes)
])
ring = ring[[13; 1:12]]
dist_between_rings = 2

layout = Vector{CM.Point2}(undef, size(pisa_data, 1))
c = 1

for i in axes(summary_df, 1)
    no_countries = length(summary_df.countries[i])
    inner_circle_xy = get_xy(no_countries) ./ 7
    inner_circle = swap_last_to_first(CM.Point2.(inner_circle_xy[1], inner_circle_xy[2]))

    j = ring_idx[i]
    k = i - sum(view(ringsizes, 1:j-1); init = 0)
    idx = mod1(k, ringsizes[j])
    multiplier = 1 / j
    center = rings[j][idx] .* (multiplier * dist_between_rings)

    for j in eachindex(inner_circle)
        layout[c] = inner_circle[j] .+ center
        c += 1
    end
end


# CM.scatter(get_xy(13)...)

# layout = NetworkLayout.spring(mpm_graph, C = 3.3, seed = 42)



ucolors = [col for (col, _) in zip(Iterators.cycle(Colors.distinguishable_colors(8)), 1:26)]
# Create legend entries
colors = ucolors[mpm_partition2]

labels = [CM.rich(string(i, ". ", replace(mpm_partition_countries[i], "_" => " ")), color = colors[i]) for i in eachindex(mpm_partition_countries)]

legend_elements = [CM.MarkerElement(marker = :circle, markersize = 0) for label in labels]


fig = CM.Figure()
# gl = fig[1, 2] = CM.GridLayout()
# ax = CM.Axis(gl[1, 1])
ax = CM.Axis(fig[1, 1])
CM.hidespines!(ax); CM.hidedecorations!(ax)
GraphMakie.graphplot!(ax, mpm_graph,
    ilabels = string.(eachindex(mpm_partition)),
    ilabels_fontsize = 20,
    node_size = 30,
    node_color = colors,
    node_attr = (;alpha = .6, strokecolor = :grey, strokewidth = .5),
    edge_color = (:grey, .1),

    layout = _ -> layout
)
fig

# Create the legend
legend = CM.Legend(
    fig[1, 2],
    # gl[1, 2],
    legend_elements,
    labels,
    # "Legend",
    # labelcolor = colors,
    nbanks = 26,
    tellwidth = false,
    tellheight = false,
    orientation = :horizontal,  # Horizontal layout for columns
)

# CM.resize!(fig, 1800, 800)
CM.colsize!(fig.layout, 2, CM.Relative(1/3))
w = 900
resize!(fig, ceil(Int, 2 * w + w/3), 800)
fig

CM.save("simulations/newfigures/pisa_mpm_network.pdf", fig)




import LaTeXTabulars, Printf
summary_df2 = copy(summary_df)
summary_df2.countries = map(x -> join(x, ", "), summary_df.countries)
summary_df2.posterior_mean = round_2_decimals.(summary_df.posterior_mean)
summary_df2.lower_cri = round_2_decimals.(summary_df.lower_cri)
summary_df2.upper_cri = round_2_decimals.(summary_df.upper_cri)
summary_df2.point_ests = map(x -> join(Int.(x), ", "), summary_df.point_ests)
tb = LaTeXTabulars.latex_tabular(String, LaTeXTabulars.Tabular("ll"),
    [
        LaTeXTabulars.Rule(:top),
        ["Countries", "Posterior Mean", "Lower CRI", "Upper CRI", "Point Estimates"],
        LaTeXTabulars.Rule(:mid),
        [collect(row) for row in eachrow(summary_df2)],
        LaTeXTabulars.Rule(:bottom)
    ]
)
print(tb)
sort!(summary_df, [:posterior_mean, :lower_cri, :upper_cri], rev = true)
showall(summary_df)


=#