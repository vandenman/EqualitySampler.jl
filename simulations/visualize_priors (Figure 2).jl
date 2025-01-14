using EqualitySampler, Distributions
import  DataFrames          as DF,
        StatsBase           as SB,
        AlgebraOfGraphics   as AoG,
        CairoMakie          as CM,
        Colors,
        OrderedCollections,
        MLStyle,
        ProgressMeter

include("priors_plot_colors_shapes_labels_new.jl")
include("plot_partitions (Figure 1).jl")

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

nmodels = length(models)

results = DF.DataFrame(
    prior     = Vector{Symbol}(         undef, nmodels),
    pdf_model = Vector{Vector{Float64}}(undef, nmodels),
    pdf_incl  = Vector{Vector{Float64}}(undef, nmodels)
)

k = 5

modelspace = collect(PartitionSpace(k))
sorted_models = sort!.(EqualitySampler.fast_countmap_partition.(modelspace))
unique_modeltypes = unique(sorted_models)
unique_modeltypes
first_idx = indexin(unique_modeltypes, sorted_models)
all_idx   = indexin(sorted_models, unique_modeltypes)

all_idx2 = [Int[] for _ in eachindex(unique_modeltypes)]
for i in eachindex(all_idx)
    push!(all_idx2[all_idx[i]], i)
end

ProgressMeter.@showprogress for (i, (name, model)) in enumerate(models)
    results.prior[i] = name
    d = model(k)
    results.pdf_model[i] = logpdf_model_distinct.(Ref(d), view(modelspace, first_idx))
    results.pdf_incl[i]  = logpdf_incl.(Ref(d), 1:k)
end

results.column = map(get_family_from_prior, results.prior)

function superimpose_networks!(ax, models; width = 1.1, y_start = -7.75, kwargs...)

    for j in eachindex(models)
        bbox = CM.lift(ax.scene.camera.projectionview, ax.scene.viewport) do _, pxa
            # width = 1.1
            x_start = j - width / 2#0.6
            y_start = y_start
            x_end = x_start + width
            y_end = y_start + width

            bl = CM.Makie.project(ax.scene, CM.Point2f(x_start, y_start)) + pxa.origin
            tr = CM.Makie.project(ax.scene, CM.Point2f(x_end,   y_end))   + pxa.origin
            CM.Rect2f(bl, tr - bl)
        end
        ax_inset = CM.Axis(fig,
            bbox  = bbox,
            # backgroundcolor=Colors.Gray(0.975),
            aspect = 1,
            xautolimitmargin = (0.f0, 0.f0),
            yautolimitmargin = (0.f0, 0.f0),
        )
        CM.hidedecorations!(ax_inset)
        CM.hidespines!(ax_inset)
        plot_one_model!(ax_inset, models[j]; kwargs...)
    end

end

# AoG.pregrouped(pdf_model = results.pdf_model, prior = results.prior, column = results.column) * AoG.mapping(
#     y = :pdf_model,
#     x = AoG.direct(eachindex(first_idx)),
#     color = :prior,
#     col = :column
# ) |> AoG.draw()

# AoG.pregrouped(
#     fill(eachindex(first_idx), nmodels),
#     results.pdf_model,
#     color = results.prior,
#     marker = results.prior,
#     col = results.column
# ) * AoG.visual(CM.Scatter; markersize = 30) |> AoG.draw(; figure = (size = (2000, 2000), ))

# AoG.pregrouped(
#     fill(1:k, nmodels),
#     results.pdf_incl,
#     color = results.prior,
#     marker = results.prior,
#     col = results.column
# ) * AoG.visual(CM.Scatter; markersize = 30) |> AoG.draw(; figure = (size = (2000, 2000), ))

color_palette  = get_color_palette(results.prior)
marker_palette = get_marker_palette(results.prior)

legend_elems, legend_contents, legend_titles = get_legend_contents(color_palette, marker_palette, true)



ww = 1500
unique_colnames = unique(results.column)
unique_colnames = [:Dirichlet, :BetaBinomial, :Uniform]
new_order = [2, 3, 1]
legend_elems    = legend_elems[new_order]
legend_contents = legend_contents[new_order]
legend_titles   = legend_titles[new_order]



function one_legend(prior_name, color_palette, marker_palette)
    color = color_palette[prior_name]
    marker = marker_palette[prior_name]
    [
        CM.LineElement(color  = color, linestyle = nothing),
        CM.MarkerElement(color = color, marker = marker, markersize = 15, strokecolor = :black)
    ]
end


legends = Dict(
    u => map(x->one_legend(x, color_palette, marker_palette), results.prior[findall(==(u), results.column)])
    for u in unique_colnames
)

yticks = (0:-2:-8, string.(0:-2:-8))
ylimits = (-9, .2)

# fig = CM.with_theme(markersize = 10, linewidth = 2) do
ms = 18 # markersize
lw = 2  # linewidth

fig = CM.Figure(fontsize = 20)#size = ww .* (length(unique_colnames), 2), fontsize = 45)
# (i, u) = first(enumerate(unique_colnames))
xautolimitmargin = (0.07f0, 0.07f0)
for (i, u) in enumerate(unique_colnames)

    idx_u = findall(==(u), results.column)
    markers = [marker_palette[results.prior[j]] for j in idx_u]
    colors  = [color_palette[results.prior[j]] for j in idx_u]

    CM.Label(fig[0, i], legend_titles[i], tellwidth = false, font = :bold)

    ax = CM.Axis(fig[1, i], #limits = ((0.5, length(first_idx) + .5), (-10, 0)),
        xlabelvisible = i==2,
        xticks = (1:7, ["" for _ in 1:7]), yticks = yticks, limits = (nothing, ylimits),
        xlabel = i == 2 ? "Model type" : "", xautolimitmargin = xautolimitmargin)
    yvals = reduce(hcat, results.pdf_model[idx_u])
    # CM.series!(ax, permutedims(yvals), markersize = 30, marker = markers, color = colors)

    for j in eachindex(idx_u)
        CM.scatterlines!(ax, axes(yvals, 1), yvals[:, j], alpha = .8, color = colors[j], marker = markers[j], markersize = ms, linewidth = lw)
    end

    # CM.Legend(fig[0, i], legends[u], string.(results.prior[findall(==(u), results.column)]), position = :tr)
    # CM.axislegend(ax, legends[u], string.(results.prior[findall(==(u), results.column)]), position = :rt,
    #     framevisible = false, backgroundcolor = :transparent)

    if u != :Uniform
        CM.axislegend(ax, legend_elems[i], legend_contents[i], position = :ct,
            framevisible = false, backgroundcolor = :transparent, margin = (0, 0, 0, -5))
    end

    superimpose_networks!(ax, view(modelspace, first_idx), width = 1.3, y_start = -8.75, markersize = 7, strokewidth = 1.2)

    ax = CM.Axis(fig[2, i], xticks = (1:5, string.(0:4)), yticks = yticks, limits = (nothing, ylimits),
        xlabel = i == 2 ? "No. inequalities" : "", xautolimitmargin = xautolimitmargin)
    yvals = reduce(hcat, results.pdf_incl[idx_u])
    # CM.series!(ax, permutedims(yvals), markersize = 30, marker = markers, color = colors)
    for j in eachindex(idx_u)
        CM.scatterlines!(ax, axes(yvals, 1), yvals[:, j], color = colors[j], marker = markers[j], markersize = ms, linewidth = lw)
    end

end
# CM.Label(fig[2, :], "Model type",       tellwidth = false)
# CM.Label(fig[4, :], "No. inequalities", tellwidth = false)

CM.Label(fig[:, 0], "Log prior probability", tellheight = false, rotation = pi/2)
# CM.Legend(fig[1:2, 3], fig.content[2], "Priors")

w = 650
resize!(fig, 1300, 800)
fig

figures_dir = joinpath(pwd(), "simulations", "revision_figures")
CM.save(joinpath(figures_dir, "prior_comparison_pdfs_incl.pdf"), fig)

#=
to_plot = randn(3, 100)
fig = CM.Figure()
ax = CM.Axis(fig[1, 1])
li = [CM.lines!(ax, to_plot[i, :]) for i in axes(to_plot, 1)]
CM.axislegend(ax, li, ["One", "Two", "Three"])
fig

fig = CM.Figure()
ax = CM.Axis(fig[1, 1])
ss = CM.series!(ax, to_plot, labels = ["One", "Two", "Three"])
CM.axislegend(ax, ss, ["One", "Two", "Three"])
fig




# extract this into a function that takes an axis and view(modelsspace, first_idx)
for j in eachindex(first_idx)
    bbox = CM.lift(ax.scene.camera.projectionview, ax.scene.viewport) do _, pxa
        width = 1.1
        x_start = j - width / 2#0.6
        y_start = -9.75
        x_end = x_start + width
        y_end = y_start + width

        bl = CM.Makie.project(ax.scene, CM.Point2f(x_start, y_start)) + pxa.origin
        tr = CM.Makie.project(ax.scene, CM.Point2f(x_end,   y_end))   + pxa.origin
        CM.Rect2f(bl, tr - bl)
    end
    ax_inset = CM.Axis(fig,
        bbox  = bbox,
        # backgroundcolor=Colors.Gray(0.975),
        backgroundcolor=Colors.Gray(0.975),
        aspect = 1,
        # title = "Group-level ROC",
        # limits = limits,
        xautolimitmargin = (0.f0, 0.f0),
        yautolimitmargin = (0.f0, 0.f0),
        titlealign = :right
    )
    CM.hidedecorations!(ax_inset)
    # CM.scatter!(ax_inset, 1:k, 1:k, markersize = 30, color = :black)
    plot_one_model!(ax_inset, modelspace[first_idx[j]])
end

fig

x = [rand(10) .+ i for i in 1:3]
y = [rand(10) .+ i for i in 1:3]
z = [rand(10) .+ i for i in 1:3]
c = ["a", "b", "c"]
df_m = DF.DataFrame(
    x = x,
    y = y,
    z = z,
    c = c
)
m = AoG.pregrouped(x, y, color=c => (t -> "Type " * t ) => "Category")
AoG.draw(m)

m2 = AoG.data(AoG.Pregrouped(df_m)) * AoG.pregrouped(x, y, color=:c => (t -> "Type " * t ) => "Category")
AoG.draw(m2)

m2 = AoG.pregrouped(df_m.x, df_m.y, color=df_m.c => (t -> "Type " * t ) => "Category")
AoG.draw(m2)
=#