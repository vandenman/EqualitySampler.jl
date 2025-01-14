import CairoMakie as CM, StatsBase as SB
import Colors, ColorSchemes
import MLStyle

function get_family_from_prior(prior)
    MLStyle.@match lowercase(first(string(prior))) begin
        'u' => :Uniform
        'b' => :BetaBinomial
        'd' => :Dirichlet
        'p' => :PitmanYor
        'w' => :Westfall
        _   => throw(ArgumentError("Unknown prior: $prior"))
    end
end

function get_color_palette(prior_nms)

    # fixed for the manuscript, the code below the return is used to generate the colors
    dict = Dict{Symbol, Colors.RGB{Float64}}(
        :BetaBinomial11       => Colors.RGB{Float64}(0.301575,0.309775,0.5473),
        :BetaBinomial1k       => Colors.RGB{Float64}(0.38673749999999996,0.580175,0.796025),
        :BetaBinomial1binomk2 => Colors.RGB{Float64}(0.5041,0.9071,0.9998),
        :DirichletProcess1_0  => Colors.RGB{Float64}(0.547125,0.252425,0.2275),
        :DirichletProcessGP   => Colors.RGB{Float64}(0.7869625,0.495125,0.29835),
        :DirichletProcessDecr => Colors.RGB{Float64}(1.0,0.8126,0.4042),
        :Westfall             => Colors.RGB{Float64}(0.2771,0.4084,0.1085),
        :Westfall_uncorrected => Colors.RGB{Float64}(0.6392500000000001,0.5798,0.05265),
        :uniform              => Colors.RGB{Float64}(0.49331718750000003,0.49331718750000003,0.49331718750000003),
    )
    return Dict(
        prior => dict[prior] for prior in prior_nms
    )
#=
    prior_nms = unique(prior_nms)
    # colorschemes = Dict(
    #     :Uniform        => :gray1,
    #     :BetaBinomial   => :managua10,
    #     :Dirichlet      => :bamako10,
    #     :PitmanYor      => :lajolla10,
    #     :Westfall       => :starrynight
    # )

    # suggestions by Fabian
    # blue_colors = parse.(Colors.Colorant, ["#AEC6CF", "#1E90FF", "#0000CC"])
    # red_colors =  parse.(Colors.Colorant, ["#FFA07A", "#FF4500", "#CC0000"])


    cols_bb       = ColorSchemes.managua[range(0.65, 1.00, length = 3)]
    cols_dpp      = ColorSchemes.managua[range(0.00, 0.35, length = 3)]
    cols_westfall = ColorSchemes.bamako[range(.40, 0.70, length = 2)]
    cols = [cols_bb; cols_dpp; cols_westfall]
    l = sum(Colors.HSL(col).l for col in cols) / length(cols)
    col_uniform  = Colors.RGB(Colors.HSL(.0, .0, l))
    # [cols_bb; cols_dpp; cols_westfall; col_uniform]

    colorschemes = Dict(
        :Uniform        => ColorSchemes.ColorScheme([col_uniform]),
        :BetaBinomial   => ColorSchemes.ColorScheme(cols_bb),
        :Dirichlet      => ColorSchemes.ColorScheme(cols_dpp),
        :PitmanYor      => ColorSchemes.lajolla10,#ColorScheme([col_uniform]),
        :Westfall       => ColorSchemes.ColorScheme(cols_westfall)
    )

    indices = Dict(
        :Uniform        => 0,
        :BetaBinomial   => 0,
        :Dirichlet      => 0,
        :PitmanYor      => 0,
        :Westfall       => 0
    )

    families = get_family_from_prior.(prior_nms)
    counts = SB.countmap(families)

    all_linecolors_dict = Dict(
        begin
            fam = get_family_from_prior(prior)
            if isone(counts[fam])
                idx = 1/2
            else
                idx = indices[fam] / (counts[fam] - 1)
                indices[fam] += 1
            end
            scheme = colorschemes[fam]
            # prior => getfield(ColorSchemes, scheme)[idx]
            prior => scheme[idx]
        end
        for prior in prior_nms
    )
    return all_linecolors_dict
=#
end

__prior_to_marker_dict = Dict(
	:uniform              => :rect,
	:BetaBinomial11       => :utriangle,
	:BetaBinomialk1       => :rtriangle,
	:BetaBinomial1k       => :ltriangle,
	:BetaBinomial1binomk2 => :dtriangle,

	# :DirichletProcess0_5  => :star4,
	:DirichletProcess1_0  => :star4,
	# :DirichletProcess2_0  => :star6,
	:DirichletProcessGP   => :star6,
    :DirichletProcessDecr => :star8,

	:PitmanYorProcess0_25__0_5 => :cross,
	:PitmanYorProcess0_50__0_5 => :cross,
	:PitmanYorProcess0_75__0_5 => :cross,
	:PitmanYorProcess0_25__1_0 => :xcross,
	:PitmanYorProcess0_50__1_0 => :xcross,
	:PitmanYorProcess0_75__1_0 => :xcross,

	:Westfall             => :circle,
	:Westfall_uncorrected => :circle
)
__prior_fam_to_marker_dict = Dict(
    :Uniform        => :gray1,
    :BetaBinomial   => :triangle,
    :Dirichlet      => :star4,
    :PitmanYor      => :cross,
    :Westfall       => :circle
)

function prior_to_marker(prior_symbol)
    get!(__prior_to_marker_dict, prior_symbol) do
        __prior_fam_to_marker_dict[get_family_from_prior(prior_symbol)]
    end
end
get_marker_palette(all_priors_nms) = Dict(prior => prior_to_marker(prior) for prior in all_priors_nms)

function get_legend_contents(color_palette0, marker_palette0, skip_westfall = false, linestype_pallete = nothing, linewidth = 1)

    idx = skip_westfall ? [1, 3, 4] : collect(1:4)
    legend_elems = [
        [
            (
                isnothing(marker_palette0) ?
                [CM.LineElement(color   = color_palette0[prior], linestyle = isnothing(linestype_pallete) ? :solid : linestype_pallete[prior], linewidth = linewidth)]
                :
                [
                    CM.LineElement(color   = color_palette0[prior], linestyle = isnothing(linestype_pallete) ? :solid : linestype_pallete[prior], linewidth = linewidth),
                    CM.MarkerElement(color = color_palette0[prior], marker = marker_palette0[prior], markersize = 20)
                ]
            )
            for prior in priorgroup
        ]
        for priorgroup in (
            [:uniform],
            [:Westfall, :Westfall_uncorrected],
            [:DirichletProcess1_0, :DirichletProcessGP, :DirichletProcessDecr],
            # [:DirichletProcessGP, :DirichletProcess1_0, :DirichletProcessDecr],
            [:BetaBinomial11, :BetaBinomial1k, :BetaBinomial1binomk2],
        )[idx]
    ]

    legend_titles = ["Uniform", "Pairwise BFs", "Dirichlet", "Beta-binomial"][idx]
    legend_contents = [
        ["Uniform"],
        ["Westfall", "Uncorrected"],
        ["α = 1", "α = G&B", "α = H(K-1)⁻¹" ],
        # ["α = G&B", "α = 1", "α = H(K)⁻¹" ],
        ["α = 1, β = 1", "α = 1, β = K", "α = 1, β = binom(K, 2)"],
    ][idx]

    return legend_elems, legend_contents, legend_titles
end

function make_legend!(fig, legend_elems, legend_contents, legend_titles, row = 2, col = 3, skip_at = nothing)
    gl = fig[row, col] = CM.GridLayout(2, 2)

    for i in eachindex(legend_elems)

        j = !isnothing(skip_at) && i in skip_at ? i + 1 : i

        i1, i2 = fldmod1(j, 2)
        CM.Legend(
            gl[i1, i2],
            legend_elems[i],
            legend_contents[i],
            legend_titles[i],
            tellwidth = false, tellheight = false, framevisible = false, nbanks = 1,
            # valign = :top, halign = :center
            halign = :left, valign = :top, titlehalign = :left,
            # margin = (i2 == 2 ? -30 : 0, 0, 0, 0)
            margin = (0, 0, 0, 0)
        )
    end
end

__priors_in_manuscript = [
    :uniform,
    :BetaBinomial11, :BetaBinomial1k, :BetaBinomial1binomk2,
    :DirichletProcess1_0, :DirichletProcessGP, :DirichletProcessDecr,
    :Westfall, :Westfall_uncorrected
]

function demo_colors_and_markers(prior_nms = [
    :uniform,
    :BetaBinomial11, :BetaBinomial1k, :BetaBinomial1binomk2,
    :DirichletProcess0_5, :DirichletProcess1_0, :DirichletProcessGP,
    :PitmanYorProcess0_1__0_0, :PitmanYorProcess0_1__0_2, :PitmanYorProcess0_1__0_4, :PitmanYorProcess0_1__0_6, :PitmanYorProcess0_1__0_8, :PitmanYorProcess0_3__m0_2, :PitmanYorProcess0_3__0_0, :PitmanYorProcess0_3__0_2, :PitmanYorProcess0_3__0_4, :PitmanYorProcess0_3__0_6, :PitmanYorProcess0_5__m0_4, :PitmanYorProcess0_5__m0_2, :PitmanYorProcess0_5__0_0, :PitmanYorProcess0_5__0_2, :PitmanYorProcess0_5__0_4, :PitmanYorProcess0_7__m0_6, :PitmanYorProcess0_7__m0_4, :PitmanYorProcess0_7__m0_2, :PitmanYorProcess0_7__0_0, :PitmanYorProcess0_7__0_2, :PitmanYorProcess0_9__m0_8, :PitmanYorProcess0_9__m0_6, :PitmanYorProcess0_9__m0_4, :PitmanYorProcess0_9__m0_2, :PitmanYorProcess0_9__0_0,
    :Westfall, :Westfall_uncorrected
])

    all_linecolors_dict = get_color_palette(prior_nms)
    all_markers_dict    = get_marker_palette(prior_nms)

    families = get_family_from_prior.(prior_nms)
    ufamilies = unique(families)
    row_idx = Dict(u => 1 for u in ufamilies)

    col_idx = Dict{keytype(row_idx), valtype(row_idx)}()
    fig = CM.Figure()
    for (i, u) in enumerate(ufamilies)
        CM.Label(fig[0, i], string(u), tellwidth = false)
        col_idx[u] = i
    end


    for prior in prior_nms
        fam = get_family_from_prior(prior)
        i = row_idx[fam]
        row_idx[fam] += 1
        j = col_idx[fam]
        # CM.Box(fig[i, j], color = all_linecolors_dict[prior])

        title = replace(lowercase(string(prior)), lowercase(string(fam)) => "")
        ax = CM.Axis(fig[i, j], title = title)
        CM.hidedecorations!(ax)
        CM.hidespines!(ax)
        CM.scatter!(ax, [0], [1], color = all_linecolors_dict[prior], marker = all_markers_dict[prior], markersize = 50)
    end
    w = 200
    CM.resize!(fig, w*length(ufamilies), w * maximum(values(row_idx)))
    fig
end

# demo_colors_and_markers(__priors_in_manuscript)