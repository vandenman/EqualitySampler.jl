using EqualitySampler, JLD2, DataFrames, Chain, StatsBase
import CodecZlib
import AlgebraOfGraphics as AOG, CairoMakie as CM, ColorSchemes, Colors, LaTeXStrings
import MLStyle
import EqualitySampler
import Printf
include("utilities.jl")
include("priors_plot_colors_shapes_labels_new.jl")


function read_latest_file(results_dir, pattern)
    files = filter!(endswith(".jld2"), readdir(results_dir))
    filter!(startswith(pattern), files)

    matches = match.(r"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}", files)

    files = files[.!isnothing.(matches)]
    matches = matches[.!isnothing.(matches)]
    dates = [Dates.DateTime.(m.match, "yyyy-mm-dd_HH:MM") for m in matches]
    _, idx = findmax(dates)
    log_message("reading file: $(files[idx])")
    return jldopen(joinpath(results_dir, files[idx]))["results_df"]

end

function get_theme()
    CM.Theme(
        Lines = (
            linewidth = 4,
        )
    )
end

function get_prior_to_remove_py(upriors)
    # priors_to_remove = Set(upriors[findall(x -> !(startswith(string(x), "PitmanYorProcess") || startswith(string(x), "DirichletProcess")), upriors)])
    priors_to_keep = Set([
        :PitmanYorProcess0_1__0_0,
        :PitmanYorProcess0_1__0_8,
        :PitmanYorProcess0_3__m0_2,
        :PitmanYorProcess0_3__m0_6,
        :PitmanYorProcess0_5__m0_4,
        :PitmanYorProcess0_5__0_4,
        :PitmanYorProcess0_7__m0_6,
        :PitmanYorProcess0_7__0_2,
        :PitmanYorProcess0_9__m0_8,
        :PitmanYorProcess0_9__0_0
    ])
    dirichlet_priors = Set(upriors[findall(x -> (startswith(string(x), "DirichletProcess")), upriors)])
    priors_to_remove = setdiff(Set(upriors), union(priors_to_keep, dirichlet_priors))
    return priors_to_remove
end

function get_prior_to_remove(upriors)
    setdiff(
        upriors,
        Set([
            :uniform,
            :BetaBinomial11, :BetaBinomial1k, :BetaBinomial1binomk2,
            :DirichletProcess1_0, :DirichletProcessGP, :DirichletProcessDecr,
            :Westfall, :Westfall_uncorrected
        ])
    )
end

function create_figure_small_simulation(results_dir, figures_dir)

    results_small_df = read_latest_file(results_dir, "combined_runs_small")

    replace!(results_small_df.prior, Symbol("DirichletProcess_1.0") => :DirichletProcess1_0)

    upriors = unique(results_small_df.prior)
    priors_to_remove = get_prior_to_remove(upriors)

    reduced_results_small_df = @chain results_small_df begin
        subset(:prior => (x -> x .∉ Ref(priors_to_remove)))
        # filter(:obs_per_group => x-> x in obs_per_group_to_keep, _)
        groupby(Cols(:prior, :groups, :hypothesis))
        combine([:any_incorrect, :prop_incorrect, :α_error_prop, :β_error_prop] .=> mean, :α_error_prop => (x->mean(>(0.0), x)) => :any_α_error_prop)
        sort([order(:hypothesis, by=x->parse(Int, string(x)[2:end]), rev=true), order(:groups)])
    end

    # only one observations per combination of prior and groups and hypotheses
    @assert all(allunique(gdf.groups) for gdf in groupby(reduced_results_small_df, Cols(:hypothesis, :prior)))

    priors_to_keep = setdiff(upriors, priors_to_remove)
    color_palette0 = get_color_palette(priors_to_keep)

    marker_palette0 = get_marker_palette(priors_to_keep)
    marker_palette = collect(marker_palette0)
    color_palette  = collect(color_palette0)
    dodge_x = 0.5
    legend_elems, legend_contents, legend_titles = get_legend_contents(color_palette0, marker_palette0)

    layers = AOG.visual(CM.Lines, linestyle = :solid, alpha = .4) +
        AOG.visual(CM.Scatter; markersize = 20, alpha = .8) * AOG.mapping(marker = :prior => "")

    # layers = AOG.visual(CM.ScatterLines; markersize = 20, alpha = .4) * AOG.mapping(marker = :prior => "")

    aog_data_left = AOG.data(subset(reduced_results_small_df, :hypothesis => ByRow(==(Symbol(:p00)))))
    mapping_left = AOG.mapping(:groups => "No. groups", :any_α_error_prop => "Probability of at least one error", group = :prior => "", color = :prior => "", dodge_x = :prior => "")
    fig_left = aog_data_left * mapping_left * layers

    axis_args = (rightspinevisible = false, topspinevisible = false)
    # figure_args = (; size = (1000, 500))
    # left_panel = AOG.draw(fig_left,  AOG.scales(Marker = (; palette = marker_palette), Color = (; palette = color_palette), DodgeX = (; width = dodge_x)), axis = axis_args, figure = figure_args)

    aog_data_right = AOG.data(subset(reduced_results_small_df, :hypothesis => ByRow(==(Symbol(:p100)))))
    mapping_right = AOG.mapping(:groups => "No. groups", :β_error_prop_mean => "Proportion of errors (β)", group = :prior => "", color = :prior => "", dodge_x = :prior => "")
    fig_right = aog_data_right * mapping_right * layers

    # right_panel = AOG.draw(fig_right,  AOG.scales(Marker = (; palette = marker_palette), Color = (; palette = color_palette), DodgeX = (; width = dodge_x)), axis = axis_args, figure = figure_args)

    expand_percent = .04
    yticks_left = 0.0:.2:1.0
    min_left, max_left = extrema(yticks_left)
    expand_left = expand_percent * (max_left - min_left)
    ylimits_left = (min_left - expand_left, max_left + expand_left)

    yticks_right = 0.0:.2:1.0
    min_right, max_right = extrema(yticks_right)
    expand_right = expand_percent * (max_right - min_right)
    ylimits_right = (min_right - expand_right, max_right + expand_right)

    fig = CM.with_theme(get_theme(), fontsize = 20) do
        fig = CM.Figure()
        # gl = fig#[1, 1] = CM.GridLayout(1, 2)
        ax_left  = CM.Axis(fig[1, 1]; title = "Null model", xlabel = "No. groups", ylabel = "Probability of at least one error", yticks = yticks_left , limits = (nothing, ylimits_left))
        ax_right = CM.Axis(fig[1, 2]; title = "Full model", xlabel = "No. groups", ylabel =  "Proportion of errors (β)",         yticks = yticks_right, limits = (nothing, ylimits_right))
        AOG.draw!(ax_left,  fig_left,  AOG.scales(Marker = (; palette = marker_palette), Color = (; palette = color_palette), DodgeX = (; width = dodge_x)))
        AOG.draw!(ax_right, fig_right, AOG.scales(Marker = (; palette = marker_palette), Color = (; palette = color_palette), DodgeX = (; width = dodge_x)))

        # legends = [CM.Legend(fig, legend_elems[i], legend_contents[i], legend_titles[i],
        #     tellwidth = false, tellheight = false, framevisible = true,
        #     halign = :left, valign = :top, titlehalign = :left,
        #     margin = (7, 0, i >2 ? 10 : 0, i < 3 ? 10 : -30),
        #     titlegap = 5) for i in 1:4]
        # fig[1, 1] = legends[1]
        # fig[1, 2] = legends[2]
        # fig[3, 1] = legends[3]
        # # fig[2, 2] = legends[4]
        # fig[2, 1] = legends[4]

        ord = [3, 4, 1, 2]
        CM.Legend(fig[1, 1], legend_elems[ord], legend_contents[ord], legend_titles[ord],
            tellwidth = false, tellheight = false, framevisible = false,
            backgroundcolor = :transparent,
            halign = :left, valign = :top, titlehalign = :left,
            labelhalign = :left,
            gridshalign = :left,
            # margin = (7, 0, i >2 ? 10 : 0, i < 3 ? 10 : -30),
            titlegap = 5)

        w = 650
        CM.resize!(fig, 2w, w)
        fig
    end
    fig

    filename = "small_simulation_results.pdf"
    save(joinpath(figures_dir, filename), fig)#, pt_per_unit = 15)

end

function create_figure_big_simulation(results_dir, figures_dir)

    results_big_df = read_latest_file(results_dir, "combined_runs_big")

    replace!(results_big_df.prior, Symbol("DirichletProcess_1.0") => :DirichletProcess1_0)

    upriors = unique(results_big_df.prior)
    priors_to_remove = get_prior_to_remove(upriors)

    reduced_results = @chain results_big_df begin
        # subset(:obs_per_group => (x -> x .<= 500))
        subset(:prior => (x -> x .∉ Ref(priors_to_remove)))
        # filter(:obs_per_group => x-> x in obs_per_group_to_keep, _)
        groupby(Cols(:obs_per_group, :prior, :groups, :hypothesis))
        combine([:any_incorrect, :prop_incorrect, :α_error_prop, :β_error_prop] .=> mean, :α_error_prop => (x->mean(>(0.0), x)) => :any_α_error_prop)
        sort([order(:hypothesis, by=x->parse(Int, string(x)[2:end]), rev=true), order(:obs_per_group)])
    end
    # reduced_results_big_df = groupby(reduced_results_big_df_ungrouped, Cols(:hypothesis, :groups))

    averaged_results = @chain reduced_results begin
        groupby(Cols(:obs_per_group, :prior, :groups))
        combine([:any_α_error_prop, :β_error_prop_mean] .=> mean)
        # groupby(:groups)
    end

    # no_equality_rn1 = AOG.renamer([
    #     :p00  => "Inequalities = 0",
    #     :p25  => "Inequalities = 1",
    #     :Averaged => "Averaged",
    #     :p50  => "Inequalities = 2",
    #     :p75  => "Inequalities = 3",
    #     :p100 => "Inequalities = 4",
    # ])
    # no_equality_rn2 = AOG.renamer([
    #     :p00  => "Inequalities = 0",
    #     :p25  => "Inequalities = 1",
    #     :p50  => "Inequalities = 2",
    #     :Averaged => "Averaged",
    #     :p75  => "Inequalities = 3",
    #     :p100 => "Inequalities = 4",
    # ])
    no_equality_rn1 = Dict(
        5 => AOG.renamer([
            :p00  => "Inequalities = 0",
            :p25  => "Inequalities = 1",
            :Averaged => "Averaged",
            :p50  => "Inequalities = 2",
            :p75  => "Inequalities = 3",
            :p100 => "Inequalities = 4",
        ]),
        9 => AOG.renamer([
            :p00  => "Inequalities = 0",
            :p25  => "Inequalities = 2",
            :Averaged => "Averaged",
            :p50  => "Inequalities = 4",
            :p75  => "Inequalities = 6",
            :p100 => "Inequalities = 8",
        ])
    )
    no_equality_rn2 = Dict(
        5 => AOG.renamer([
            :p100 => "Equalities = 0",
            :p75  => "Equalities = 1",
            :Averaged => "Averaged",
            :p50  => "Equalities = 2",
            :p25  => "Equalities = 3",
            :p00  => "Equalities = 4",
        ]),
        9 => AOG.renamer([
            :p100 => "Equalities = 0",
            :p75  => "Equalities = 2",
            :Averaged => "Averaged",
            :p50  => "Equalities = 4",
            :p25  => "Equalities = 6",
            :p00  => "Equalities = 8",
        ])
    )

    axis_args = (rightspinevisible = false, topspinevisible = false)
    figure_args = (; size = (1200, 600))
    alpha = .75

    priors_to_keep = setdiff(upriors, priors_to_remove)
    color_palette0 = get_color_palette(priors_to_keep)
    marker_palette0 = get_marker_palette(priors_to_keep)
    marker_palette = collect(marker_palette0)
    color_palette  = collect(color_palette0)
    dodge_x = 15
    aog_scales = AOG.scales(Marker = (; palette = marker_palette), Color = (; palette = color_palette), DodgeX = (; width = dodge_x))

    legend_elems, legend_contents, legend_titles = get_legend_contents(color_palette0, marker_palette0)

    extra_axis_args = Dict(
        (:family, 5) => (yticks = 0 : .25 :  .75, limits =(nothing, (-.04,  .79))),
        (:family, 9) => (yticks = 0 : .20 : 1.00, limits =(nothing, (-.04, 1.04))),
        (:alpha,  5) => (yticks = 0 : .10 : 0.40, limits =(nothing, (-.02, 0.42))),
        (:alpha,  9) => (yticks = 0 : .20 : 1.00, limits =(nothing, (-.04, 1.04))),
        (:power,  5) => (yticks = 0 : .20 : 1.00, limits =(nothing, (-.04, 1.04))),
        (:power,  9) => (yticks = 0 : .20 : 1.00, limits =(nothing, (-.04, 1.04)))
    )
    # size is determined by Figure 4, which looks nice with 1200 x 600 and is 1 by 2 panels.
    # Keeping the width fixed to 1200, we now make 2 by 3 panel plots, so we end up with a height of 800

    w = 650
    unscaled = [2, 3] .* w
    scaled = round.(Int, unscaled .* (2w / maximum(unscaled)))
    theme_args = (; linewidth = 2, fontsize = 20, size = (scaled[2], scaled[1]))

    for k in unique(results_big_df.groups)

        averaged_results_α = @chain reduced_results begin
            subset(:groups => ByRow(==(k)), :hypothesis => ByRow(!(==(:p100))))
            transform!(:prior => ByRow(string))
            groupby(Cols(:obs_per_group, :prior, :groups))
            combine([:any_α_error_prop, :α_error_prop_mean] .=> mean)
            rename( :any_α_error_prop_mean => :any_α_error_prop,
                    :α_error_prop_mean_mean => :α_error_prop_mean)
            select(Not(:groups))
            transform(:obs_per_group => (x -> fill(:Averaged, length(x))) => :hypothesis)
        end

        results_α = @chain reduced_results begin
            subset(:groups => ByRow(==(k)), :hypothesis => ByRow(!(==(:p100))))
            transform!(:prior => ByRow(string))
            select(:obs_per_group, :prior, :hypothesis, :any_α_error_prop, :α_error_prop_mean)
        end

        results_α = vcat(results_α, averaged_results_α)

        averaged_results_β = @chain reduced_results begin
            subset(:groups => ByRow(==(k)), :hypothesis => ByRow(!(==(:p00))))
            transform!(:prior => ByRow(string))
            groupby(Cols(:obs_per_group, :prior, :groups))
            combine(:β_error_prop_mean .=> mean)
            rename( :β_error_prop_mean_mean => :β_error_prop_mean)
            select(Not(:groups))
            transform(:obs_per_group => (x -> fill(:Averaged, length(x))) => :hypothesis)
        end

        results_β = @chain reduced_results begin
            subset(:groups => ByRow(==(k)), :hypothesis => ByRow(!(==(:p00))))
            transform!(:prior => ByRow(string))
            select(:obs_per_group, :prior, :hypothesis, :β_error_prop_mean)
        end
        results_β = vcat(results_β, averaged_results_β)

        aog_data = AOG.data(results_α)
        mapping = AOG.mapping(:obs_per_group => "No. observations", :any_α_error_prop => "Familywise error rate", group = :prior, color = :prior, dodge_x = :prior, layout = :hypothesis => no_equality_rn1[k])
        layers =  AOG.visual(CM.Scatter; markersize = 20) * AOG.mapping(marker = :prior) + AOG.visual(CM.Lines, linestyle = :solid, alpha = .5, linewidth = 4)
        # familywise_error_plt = AOG.draw(aog_data * mapping * layers, aog_scales, axis = axis_args, figure = figure_args)

        familywise_error_plt = CM.with_theme(; theme_args...) do

            familywise_error_plt = CM.Figure(; figure_args...)
            AOG.draw!(familywise_error_plt, aog_data * mapping * layers, aog_scales, axis = merge(axis_args, extra_axis_args[(:family, k)]))
            ord = [3, 1, 4, 2]
            make_legend!(familywise_error_plt, legend_elems[ord], legend_contents[ord], legend_titles[ord])
            familywise_error_plt
        end
        resize!(familywise_error_plt, scaled[2], scaled[1])
        familywise_error_plt

        aog_data = AOG.data(results_α)
        mapping = AOG.mapping(:obs_per_group => "No. observations", :α_error_prop_mean => "Proportion of errors (α)", group = :prior, color = :prior, dodge_x = :prior, layout = :hypothesis => no_equality_rn1[k])
        # layers =  AOG.visual(CM.Scatter) * AOG.mapping(marker = :prior) + AOG.visual(CM.Lines, linestyle = :solid, alpha = .5)
        # prop_error_plt = AOG.draw(aog_data * mapping * layers, aog_scales, axis = axis_args, figure = figure_args)

        prop_error_plt = CM.with_theme(; theme_args...) do
            prop_error_plt = CM.Figure(; figure_args...)
            AOG.draw!(prop_error_plt, aog_data * mapping * layers, aog_scales, axis = merge(axis_args, extra_axis_args[(:alpha, k)]))
            ord = [3, 1, 4, 2]
            make_legend!(prop_error_plt, legend_elems[ord], legend_contents[ord], legend_titles[ord])
            prop_error_plt
        end
        resize!(prop_error_plt, scaled[2], scaled[1])
        prop_error_plt

        aog_data = AOG.data(results_β)
        mapping = AOG.mapping(:obs_per_group => "No. observations", :β_error_prop_mean => "Proportion of errors (β)", group = :prior, color = :prior, dodge_x = :prior, layout = :hypothesis => no_equality_rn2[k])
        # layers =  AOG.visual(CM.Scatter) * AOG.mapping(marker = :prior) + AOG.visual(CM.Lines, linestyle = :solid, alpha = .5)
        # power_plt = AOG.draw(aog_data * mapping * layers, aog_scales, axis = axis_args, figure = figure_args)

        power_plt = CM.with_theme(; theme_args...) do
            power_plt = CM.Figure(; figure_args...)
            AOG.draw!(power_plt, aog_data * mapping * layers, aog_scales, axis = merge(axis_args, extra_axis_args[(:power, k)]))
            ord = [3, 1, 4, 2]
            make_legend!(power_plt, legend_elems[ord], legend_contents[ord], legend_titles[ord])
            power_plt
        end
        resize!(power_plt, scaled[2], scaled[1])
        power_plt

        AOG.save(joinpath(figures_dir, "familywise_error_k=$k.pdf"), familywise_error_plt)#,   pt_per_unit = 5)
        AOG.save(joinpath(figures_dir, "prop_error.pdf_k=$k.pdf"),   prop_error_plt)#,         pt_per_unit = 15)
        AOG.save(joinpath(figures_dir, "power_k=$k.pdf"),            power_plt)#,              pt_per_unit = 15)
    end
end

function create_figure_pitman_yor_simulation(results_dir, figures_dir)

    results_py_dp_df = read_latest_file(results_dir, "combined_runs_pitmanyor_dirichlet")


    expected_avg_rows = length(unique(results_py_dp_df.obs_per_group)) *
        length(unique(results_py_dp_df.prior)) *
        length(unique(results_py_dp_df.groups))

    upriors = unique(results_py_dp_df.prior)
    priors_to_remove = get_prior_to_remove_py(upriors)

    λ(x, i) = length(x) >= i ? x[i] : nothing
    results_py_dp_df.θ = map(x->λ(reverse(x), 1), results_py_dp_df.prior_args)
    results_py_dp_df.d = map(x->λ(reverse(x), 2), results_py_dp_df.prior_args)

    for i in eachindex(results_py_dp_df.θ)
        if isnothing(results_py_dp_df.θ[i])
            if results_py_dp_df.prior[i] == :DirichletProcessGP
                results_py_dp_df.θ[i] = DirichletProcessPartitionDistribution(results_py_dp_df.groups[i], :Gopalan_Berry).α
            elseif results_py_dp_df.prior[i] == :DirichletProcessDecr
                results_py_dp_df.θ[i] = DirichletProcessPartitionDistribution(results_py_dp_df.groups[i], :harmonic).α
            else
                throw(error("Unknown prior $(results_py_dp_df.prior[i])"))
            end
        end
    end
    results_py_dp_df.θ = Vector{Float64}(results_py_dp_df.θ)

    reduced_results = @chain results_py_dp_df begin
        groupby(Cols(:obs_per_group, :prior, :groups, :hypothesis))
        combine([:any_incorrect, :prop_incorrect, :α_error_prop, :β_error_prop] .=> mean, :α_error_prop => (x->mean(>(0.0), x)) => :any_α_error_prop,
            :family => first => :family, :θ => first => :θ, :d => first => :d)
        sort([order(:hypothesis, by=x->parse(Int, string(x)[2:end]), rev=true), order(:obs_per_group)])
    end
    # reduced_results_big_df = groupby(reduced_results_big_df_ungrouped, Cols(:hypothesis, :groups))

    averaged_results = @chain reduced_results begin
        groupby(Cols(:obs_per_group, :prior, :groups))
        # dividing by .8 is a hack, computing the mean of any_α_error_prop should account for the fact p100 cannot make any Type-I errors
        # and p00 cannot make any Type-II errors.
        combine([:any_α_error_prop, :β_error_prop_mean] .=> (x -> mean(x) / .8) .=> [:any_α_error_prop_mean, :β_error_prop_mean_mean],
            :family => first => :family, :θ => first => :θ, :d => first => :d)
    end

    temp = @chain averaged_results begin
        subset(:family => ByRow(==(:PitmanYorProcess)))
    end
    unique_ds = unique(temp.d)
    d_idx = [1, 4, 7, 10, 13, 16, 19, 22, 25]

    # originally used to simulate data
    αs = logrange(1e-3, 1e1, 25)
    ds = range(1e-5, 1 - 1e-5, 25)
    θs = logrange(1e-3, 1e1, 25)

    ds_rational = range(1 // 100_000, 1 - 1 // 100_000, 25)
    ds_to_plot = ds_rational[d_idx]
    ds_to_plot = Vector{Union{Rational,Float64}}(Rational.(round.(ds[d_idx], digits = 3)))
    ds_to_plot[1]   = 1e-5
    ds_to_plot[end] = 1 - 1e-5
    d_syms = Vector{LaTeXStrings.LaTeXString}(undef, length(ds_to_plot))
    for i in eachindex(ds_to_plot)
        if i  == 1
            d_syms[i] = CM.L"10^{-5}"
        elseif i == length(ds_to_plot)
            d_syms[i] = CM.L"1 - 10^{-5}"
        else
            d_syms[i] = CM.L"\frac{%$(numerator(ds_to_plot[i]))}{%$(denominator(ds_to_plot[i]))}"
        end
    end
    # Map for Unicode fraction characters
    fraction_map = Dict(
        1//8 => '⅛', 2//8 => '¼', 3//8 => '⅜', 4//8 => '½',
        5//8 => '⅝', 6//8 => '¾', 7//8 => '⅞'
    )
    d_syms = Vector{String}(undef, length(ds_to_plot))
    for i in eachindex(ds_to_plot)
        if i  == 1
            d_syms[i] = "10⁻⁵"
        elseif i == length(ds_to_plot)
            d_syms[i] = "1 - 10⁻⁵"
        else
            d_syms[i] = string(fraction_map[ds_to_plot[i]])
        end
    end


    # d_vals = unique_ds[d_idx]
    # [10^v for v in -3:.5:1]
    # d_syms = [Symbol("10^$(v)") for v in -3:.5:1]
    # d_syms = [CM.L"10^{%$v}" for v in -3:.5:1]

    for i in eachindex(averaged_results.d)
        if isnothing(averaged_results.d[i]) && averaged_results.family[i] == :DirichletProcess
            averaged_results.d[i] = 0.0
        end
    end
    averaged_results.d = Vector{Float64}(averaged_results.d)

    averaged_results.d_category = map(eachrow(averaged_results)) do row
        if row.family === :DirichletProcess
            # return CM.L"0"#Symbol(0)
            return "0"
        elseif row.d in d_vals
            return d_syms[findfirst(isequal(row.d), d_vals)]
        else
            # return :remove
            return "remove"
        end
    end

    averaged_results_subset = @chain averaged_results begin
        # subset(:d_category => ByRow(!=(:remove)))
        subset(:d_category => ByRow(!=("remove")))
    end
    # averaged_results_subset.d_category = Vector{LaTeXStrings.LaTeXString}(averaged_results_subset.d_category)
    averaged_results_subset.d_category = Vector{String}(averaged_results_subset.d_category)
    layers =
        AOG.visual(CM.Scatter; markersize = 20, alpha = .65) *
        AOG.mapping(marker = :family => AOG.renamer([
            :DirichletProcess => "Dirichlet process"
            :PitmanYorProcess => "Pitman-Yor process"
        ]) => "Prior family") +
        AOG.visual(CM.Lines, linestyle = :solid, alpha = .25)

    averaged_results_subset2 = sort(averaged_results_subset, [:family, :d, :θ], rev = false)


    # TODO: color gradient here!
    unique_d = unique(averaged_results_subset2.d)
    colors_d = ColorSchemes.viridis[range(0, 1, length = length(unique_d))]
    colors_d[1] = Colors.RGB(0.0, 0.0, 0.0)
    dodge_x = .025
    dodge_y = .005
    aog_scales = AOG.scales(
        DodgeX = (; width = dodge_x),
        DodgeY = (; width = dodge_y),
        Group = (; categories = unique(averaged_results_subset2.d_category)[sortperm(unique(averaged_results_subset2.d))]),
        Color = (; palette = colors_d, categories = unique(averaged_results_subset2.d_category)[sortperm(unique(averaged_results_subset2.d))])
    )

    aog_data = AOG.data(subset(averaged_results_subset2, :prior => x-> x .!= :DirichletProcessGP))
    mapping_left = AOG.mapping(:θ => "θ, α", :any_α_error_prop_mean => "Probability of at least one error", group = :d_category => "Discount parameter", color = :d_category => "Discount parameter", dodge_x = :d_category, dodge_y = :d_category)
    fig_left = aog_data * mapping_left * layers
    AOG.draw(fig_left, aog_scales)

    mapping_right = AOG.mapping(:θ => "θ, α", :β_error_prop_mean_mean => "Proportion of errors (β)", group = :d_category => "Discount parameter", color = :d_category => "Discount parameter", dodge_x = :d_category, dodge_y = :d_category)
    fig_right = aog_data * mapping_right * layers
    AOG.draw(fig_right, aog_scales)

    xticks = 0:2:10
    yticks = 0:.2:1


    fig = CM.with_theme(get_theme(), fontsize = 20) do
        fig = CM.Figure()
        # CM.Label(fig[2, 1:2], "θ, α", tellwidth = false, tellheight = true)
        ax_left  = CM.Axis(fig[1, 1]; xlabel = "α", xticks = xticks, yticks = yticks, ylabel = "Probability of at least one error")#, #=yticks = 0.0:.25:.75, limits = (nothing, (-0.04, .75)),=# axis_args...)
        ax_right = CM.Axis(fig[1, 2]; xlabel = "α", xticks = xticks, yticks = yticks, ylabel =  "Proportion of errors (β)")#,         #=xyticks = 0.0:.01:.06, limits = (nothing, (-0.04, .06)),=# axis_args...)
        temp = AOG.draw!(ax_left,  fig_left,  aog_scales)#,  AOG.scales(Marker = (; palette = marker_palette), Color = (; palette = color_palette), DodgeX = (; width = dodge_x)))
            AOG.draw!(ax_right, fig_right, aog_scales)#, AOG.scales(Marker = (; palette = marker_palette), Color = (; palette = color_palette), DodgeX = (; width = dodge_x)))
            AOG.legend!(fig[1, 2], temp; tellheight=false, tellwidth=false, halign=:right, valign=:top, framevisible = false)#, nbanks = 7,
        fig
    end
    w = 650
    resize!(fig, 2w, w)
    fig

    AOG.save(joinpath(figures_dir, "pitman_yor_vs_dpp_k=5.pdf"), fig)

end

function main(; results_dir, figures_dir)
    !isdir(figures_dir) && mkpath(figures_dir)
    create_figure_small_simulation(results_dir, figures_dir)
    create_figure_big_simulation(  results_dir, figures_dir)
    create_figure_pitman_yor_simulation(results_dir, figures_dir)

end

main(
    results_dir = joinpath(pwd(), "simulations", "saved_objects"),
    figures_dir = joinpath(pwd(), "simulations", "revision_figures")
)

#=

result_dir = "simulations/saved_objects/"

results_small_df = read_latest_file(result_dir, "combined_runs_small_")

for i in axes(results_small_df, 1)
    results_small_df.hypothesis[i] = if results_small_df.hypothesis[i] == :full
        :p100
    else
        :p00
    end
end

reduced_results_small_df = @chain results_small_df begin
    subset(:prior => (x -> x .∉ Ref(priors_to_remove)))
	# filter(:obs_per_group => x-> x in obs_per_group_to_keep, _)
	groupby(Cols(:prior, :groups, :hypothesis))
	combine([:any_incorrect, :prop_incorrect, :α_error_prop, :β_error_prop] .=> mean, :α_error_prop => (x->mean(>(0.0), x)) => :any_α_error_prop)
	sort([order(:hypothesis, by=x->parse(Int, string(x)[2:end]), rev=true), order(:groups)])
end

# only one observations per combination of prior and groups and hypotheses
@assert all(allunique(gdf.groups) for gdf in groupby(reduced_results_small_df, Cols(:hypothesis, :prior)))

aog_data = AOG.data(subset(reduced_results_small_df, :hypothesis => ByRow(==(Symbol(:p00)))))
mapping = AOG.mapping(:groups => "No. groups", :any_α_error_prop => "Probability of at least one error", group = :prior => "", color = :prior => "")
layers =  AOG.visual(Scatter) * AOG.mapping(marker = :prior => "") + AOG.visual(Lines, linestyle = :solid, alpha = .5)
fig_left = aog_data * mapping * layers

left_panel = AOG.draw(fig_left,  AOG.scales(Marker = (; palette = marker_palette), Color = (; palette = color_palette)), axis = axis_args, figure = figure_args)

aog_data = AOG.data(subset(reduced_results_small_df, :hypothesis => ByRow(==(Symbol(:p100)))))
mapping = AOG.mapping(:groups => "No. groups", :β_error_prop_mean => "Proportion of errors (β)", group = :prior => "", color = :prior => "")
layers =  AOG.visual(Scatter) * AOG.mapping(marker = :prior => "") + AOG.visual(Lines, linestyle = :solid, alpha = .5)
fig_right = aog_data * mapping * layers
right_panel = AOG.draw(fig_right,  AOG.scales(Marker = (; palette = marker_palette), Color = (; palette = color_palette)), axis = axis_args, figure = figure_args)

fig = CM.Figure()
gl = fig#[1, 1] = CM.GridLayout(1, 2)
ax_left  = CM.Axis(fig[1, 1]; title = "Null model", ylabel = "Probability of at least one error", axis_args...)
ax_right = CM.Axis(fig[1, 2]; title = "Full model", ylabel =  "Proportion of errors (β)", axis_args...)
temp = AOG.draw!(ax_left, fig_left,  AOG.scales(Marker = (; palette = marker_palette), Color = (; palette = color_palette)))
       AOG.draw!(ax_right, fig_right,  AOG.scales(Marker = (; palette = marker_palette), Color = (; palette = color_palette)))
AOG.legend!(fig[1, 1], temp; halign=:left, valign=:top, backgroundcolor=:transparent, framevisible=false, tellwidth=false, tellheight=false)
# CM.colsize!(fig.layout, 1, CM.Relative(3/4))
CM.resize!(fig, 1000, 500)
fig

results_df = read_latest_file(result_dir, "combined_runs_big_")

priors_to_remove = Set((
    :BetaBinomialk1,
    :DirichletProcess2_0,
    # :PitmanYorProcess0_25__0_5,
    # :PitmanYorProcess0_50__0_5,
    # :PitmanYorProcess0_75__0_5,
    # :PitmanYorProcess0_25__1_0,
    # :PitmanYorProcess0_50__1_0,
    # :PitmanYorProcess0_75__1_0
))

reduced_results_df_ungrouped = @chain results_df begin
	# subset(:obs_per_group => (x -> x .<= 500))
	subset(:prior => (x -> x .∉ Ref(priors_to_remove)))
	# filter(:obs_per_group => x-> x in obs_per_group_to_keep, _)
	groupby(Cols(:obs_per_group, :prior, :groups, :hypothesis))
	combine([:any_incorrect, :prop_incorrect, :α_error_prop, :β_error_prop] .=> mean, :α_error_prop => (x->mean(>(0.0), x)) => :any_α_error_prop)
	sort([order(:hypothesis, by=x->parse(Int, string(x)[2:end]), rev=true), order(:obs_per_group)])
end
reduced_results_df = groupby(reduced_results_df_ungrouped, Cols(:hypothesis, :groups))

reduced_results_averaged_df = @chain reduced_results_df_ungrouped begin
	groupby(Cols(:obs_per_group, :prior, :groups))
	combine([:any_α_error_prop, :β_error_prop_mean] .=> mean)
	# groupby(:groups)
end

reduced_results_averaged_df_2 = @chain reduced_results_df_ungrouped begin
	subset(:groups => ByRow(==(5)), :hypothesis => ByRow(!(==(:p100))))
	transform!(:prior => ByRow(string))
	groupby(Cols(:obs_per_group, :prior, :groups))
	combine([:any_α_error_prop, :α_error_prop_mean] .=> mean)
    rename( :any_α_error_prop_mean => :any_α_error_prop,
            :α_error_prop_mean_mean => :α_error_prop_mean)
    select(Not(:groups))
    transform(:obs_per_group => (x -> fill(:Averaged, length(x))) => :hypothesis)
end

reduced_results_df_ungrouped_2 = @chain reduced_results_df_ungrouped begin
	subset(:groups => ByRow(==(5)), :hypothesis => ByRow(!(==(:p100))))
	transform!(:prior => ByRow(string))
    select(:obs_per_group, :prior, :hypothesis, :any_α_error_prop, :α_error_prop_mean)
end

reduced_results_df_ungrouped_2 = vcat(reduced_results_df_ungrouped_2, reduced_results_averaged_df_2)

reduced_results_averaged_df_3 = @chain reduced_results_df_ungrouped begin
	subset(:groups => ByRow(==(5)), :hypothesis => ByRow(!(==(:p00))))
	transform!(:prior => ByRow(string))
	groupby(Cols(:obs_per_group, :prior, :groups))
	combine(:β_error_prop_mean .=> mean)
    rename( :β_error_prop_mean_mean => :β_error_prop_mean)
    select(Not(:groups))
    transform(:obs_per_group => (x -> fill(:Averaged, length(x))) => :hypothesis)
end
reduced_results_df_ungrouped_3 = @chain reduced_results_df_ungrouped begin
	subset(:groups => ByRow(==(5)), :hypothesis => ByRow(!(==(:p00))))
	transform!(:prior => ByRow(string))
    select(:obs_per_group, :prior, :hypothesis, :β_error_prop_mean)
end
reduced_results_df_ungrouped_3 = vcat(reduced_results_df_ungrouped_3, reduced_results_averaged_df_3)


showall(unique(reduced_results_df_ungrouped_2.prior))
marker_palette = [
	:uniform              => :rect,
	:BetaBinomial11       => :utriangle,
	:BetaBinomialk1       => :rtriangle,
	:BetaBinomial1k       => :ltriangle,
	:BetaBinomial1binomk2 => :dtriangle,

	:DirichletProcess0_5  => :star4,
	:DirichletProcess1_0  => :star5,
	:DirichletProcess2_0  => :star6,
	:DirichletProcessGP   => :star8,

	:PitmanYorProcess0_25__0_5 => :cross,
	:PitmanYorProcess0_50__0_5 => :cross,
	:PitmanYorProcess0_75__0_5 => :cross,
	:PitmanYorProcess0_25__1_0 => :xcross,
	:PitmanYorProcess0_50__1_0 => :xcross,
	:PitmanYorProcess0_75__1_0 => :xcross,

	:Westfall             => :circle,
	:Westfall_uncorrected => :circle,

	:rect
]

alpha = .75
# colors = ColorSchemes.alphacolor.(ColorSchemes.seaborn_colorblind[1:10], alpha)
colors = Colors.distinguishable_colors(18)[2:end]
color_palette = [
		:uniform              => colors[1],

		:BetaBinomial11       => colors[2],
		:BetaBinomialk1       => colors[3],
		:BetaBinomial1k       => colors[4],
		:BetaBinomial1binomk2 => colors[5],

		:DirichletProcess0_5  => colors[6],
		:DirichletProcess1_0  => colors[7],
		:DirichletProcess2_0  => colors[8],
		:DirichletProcessGP   => colors[9],

		:Westfall             => colors[10],
		:Westfall_uncorrected => colors[11],
        # :Westfall_uncorrected => colors[3],

        :PitmanYorProcess0_25__0_5 => colors[12],
        :PitmanYorProcess0_50__0_5 => colors[13],
        :PitmanYorProcess0_75__0_5 => colors[14],
        :PitmanYorProcess0_25__1_0 => colors[15],
        :PitmanYorProcess0_50__1_0 => colors[16],
        :PitmanYorProcess0_75__1_0 => colors[17],


		:black
]

no_equality_rn1 = AOG.renamer([
    :p00  => "Inequalities = 0",
    :p25  => "Inequalities = 1",
    :Averaged => "Averaged",
    :p50  => "Inequalities = 2",
    :p75  => "Inequalities = 3",
    :p100 => "Inequalities = 4",
])
no_equality_rn2 = AOG.renamer([
    :p00  => "Inequalities = 0",
    :p25  => "Inequalities = 1",
    :p50  => "Inequalities = 2",
    :Averaged => "Averaged",
    :p75  => "Inequalities = 3",
    :p100 => "Inequalities = 4",
])

axis_args = (rightspinevisible = false, topspinevisible = false)
figure_args = (; size = (1000, 500))



# reduced_results_df_ungrouped_2.prior = string.(reduced_results_df_ungrouped_2.prior)
# aog_data = AOG.data(reduced_results_df_ungrouped_2)
# mapping = AOG.mapping(:obs_per_group, :any_α_error_prop, group = :prior, color = :prior, layout = :hypothesis, marker = :prior)
# layers =  AOG.visual(Makie.ScatterLines, linestyle = :solid)#AOG.visual(Scatter) * AOG.mapping(marker = :prior) + AOG.visual(Lines, linestyle = :dash)

# custom_layout = (; palette = [(1, 1), (2, 1), (1, 2), (2, 2), (3//2, 3)])

# or Makie.ScatterLines?
aog_data = AOG.data(reduced_results_df_ungrouped_2)
mapping = AOG.mapping(:obs_per_group => "No. observations", :any_α_error_prop => "Familywise error rate", group = :prior, color = :prior, layout = :hypothesis => no_equality_rn1)
layers =  AOG.visual(Scatter) * AOG.mapping(marker = :prior) + AOG.visual(Lines, linestyle = :solid, alpha = .5)
familywise_error_plt = AOG.draw(aog_data * mapping * layers,  AOG.scales(Marker = (; palette = marker_palette), Color = (; palette = color_palette)), axis = axis_args, figure = figure_args)

aog_data = AOG.data(reduced_results_df_ungrouped_2)
mapping = AOG.mapping(:obs_per_group => "No. observations", :α_error_prop_mean => "Proportion of errors (α)", group = :prior, color = :prior, layout = :hypothesis => no_equality_rn1)
layers =  AOG.visual(Scatter) * AOG.mapping(marker = :prior) + AOG.visual(Lines, linestyle = :solid, alpha = .5)
prop_error_plt = AOG.draw(aog_data * mapping * layers,  AOG.scales(Marker = (; palette = marker_palette), Color = (; palette = color_palette)), axis = axis_args, figure = figure_args)

aog_data = AOG.data(reduced_results_df_ungrouped_3)
mapping = AOG.mapping(:obs_per_group => "No. observations", :β_error_prop_mean => "Proportion of errors (β)", group = :prior, color = :prior, layout = :hypothesis => no_equality_rn2)
layers =  AOG.visual(Scatter) * AOG.mapping(marker = :prior) + AOG.visual(Lines, linestyle = :solid, alpha = .5)
power_plt = AOG.draw(aog_data * mapping * layers,  AOG.scales(Marker = (; palette = marker_palette), Color = (; palette = color_palette)), axis = axis_args, figure = figure_args)

save("simulations/newfigures/familywise_error.pdf", familywise_error_plt, 	px_per_unit = 15)
save("simulations/newfigures/prop_error.pdf", 		prop_error_plt, 		px_per_unit = 15)
save("simulations/newfigures/power.pdf", 			power_plt, 				px_per_unit = 15)


#=
results_df5 = subset(subset, :groups => ByRow(==(5)))
results_df5.true_model[1]
results_df5.post_probs[1]

results_df5_bb1ck2 = subset(results_df5, :prior => ByRow(==(:BetaBinomial1binomk2)))

prop_incorrect_αβ.(results_df5_bb1ck2.post_probs, results_df5_bb1ck2.true_model)

# validates the way the simulation is run
import Random, EqualitySampler
rng = Random.default_rng()
log_prior_probs = instantiate_log_prior_probs_obj(5)

dat, _, true_model, _ = simulate_data_one_run(rng, :p00, 5, 100, .2)

post_probs = fit_one_run(dat, log_prior_probs[5])

y = dat.y
g = dat.g
no_groups = length(g)
gcollected = reduce(vcat, [fill(i, length(g[i])) for i in eachindex(g)])

replicate = map(instantiate_priors(5)) do prior
    results  = anova_enumerate(y, gcollected, prior, verbose = false, useBrob = true)
    log_model_probs_to_equality_probs(5, results.log_posterior_probs)
end
replicate2 = reduce((x, y) -> cat(x, y; dims = 3), replicate)
result_westfall = EqualitySampler.Simulations.westfall_test(dat)
replicate21 = cat(replicate2,  result_westfall.log_posterior_odds_mat, dims = 3)
replicate22 = cat(replicate21, result_westfall.logbf_matrix, dims = 3)
post_probs - replicate22

=#

=#