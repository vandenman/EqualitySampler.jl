using EqualitySampler, EqualitySampler.Simulations, Plots, Plots.PlotMeasures
using Turing
import  DataFrames      as DF,
        LinearAlgebra   as LA,
        NamedArrays     as NA,
        CSV,
        AbstractMCMC,
        KernelDensity,
        ColorSchemes,
        Printf
import Chain: @chain

round_2_decimals(x::Number) = Printf.@sprintf "%.2f" x
round_2_decimals(x) = x


#region Turing models
@model function variances_full_model(s, n, αs = fill(one(eltype(s)), length(s)), partition = nothing, ::Type{T} = Float64) where {T}

    k = length(s)
    v = n .- 1

    τ̄  ~ JeffreysPriorVariance()
    ϑᵘ ~ Dirichlet(αs)

    ϑᶜ = isnothing(partition) ? ϑᵘ : average_equality_constraints(ϑᵘ, partition)

    τ = ϑᶜ .* (τ̄ * k)

    # Equation B.3 (the log posterior density with μ integrated out)
    Turing.@addlogprob! - (
        sum(v) / 2 * log(2π) +
        sum(log, n) / 2 -
        sum(i->v[i] / 2 * log(τ[i]), eachindex(τ)) +
        1/2 * sum(i->v[i] * s[i] * τ[i], eachindex(τ))
    )

    return ϑᶜ, τ

end

@model function variances_equality_model(s, n, partition_prior::D, αs = fill(one(eltype(s)), length(s)), ::Type{T} = Float64) where {T, D<:AbstractPartitionDistribution}

    partition ~ partition_prior
    DynamicPPL.@submodel prefix=false ϑᶜ, τ = variances_full_model(s, n, αs, partition, T)
    return ϑᶜ, τ

end
#endregion

#region Example with simulated data
s = Float64[5, 2, 3, 3, 2]
τ = 1 ./ s
n = fill(1000, length(s))

full_mod = variances_full_model(s, n)
full_chn = sample(full_mod, NUTS(), 10_000)

full_gen = DynamicPPL.generated_quantities(full_mod, MCMCChains.get_sections(full_chn, :parameters))
full_τ_chn = reduce(vcat, permutedims.(last.(full_gen)))
full_τ_est = vec(mean(full_τ_chn; dims = 1))
isapprox(full_τ_est, τ; atol = .01)

eqs_mod = variances_equality_model(s, n, BetaBinomialPartitionDistribution(5, 1, 5))
eqs_spl = Simulations.get_sampler(eqs_mod)
eqs_chn = sample(eqs_mod, eqs_spl, 10_000)

eqs_gen = DynamicPPL.generated_quantities(eqs_mod, MCMCChains.get_sections(eqs_chn, :parameters))
eqs_τ_chn = reduce(vcat, permutedims.(last.(eqs_gen)))
eqs_τ_est = vec(mean(eqs_τ_chn; dims = 1))
isapprox(eqs_τ_est, τ; atol = .01)

partition_samples = Int.(Array(group(eqs_chn, :partition)))
eqs_mat = compute_post_prob_eq(partition_samples)

NA.NamedArray(
    LA.UnitLowerTriangular(round.(eqs_mat; digits = 3)),
    (string.(eachindex(s)), string.(eachindex(s)))
)


#endregion

#region Example with peer-rated conscientious ratings of the Czech, Estonian, German, and Flemish population

data = DF.DataFrame(CSV.File(joinpath("simulations", "analysisLevenes", "country_differences.csv")))

data_reduced = @chain data begin
    DF.groupby(DF.Cols(:country))
    DF.combine(
        :conscientiousness => (x->var(x; corrected = false)) => :s,
        :conscientiousness => length => :n
    )
end

data_reduced.abbreviation = ["German", "Flemish", "Czech", "Estonian"]

s, n = data_reduced.s, data_reduced.n
k = length(s)
τ_obs = 1 ./ s

full_mod = variances_full_model(s, n)
full_chn = sample(full_mod, NUTS(), 10_000)

full_gen = DynamicPPL.generated_quantities(full_mod, MCMCChains.get_sections(full_chn, :parameters))
full_τ_chn = reduce(vcat, permutedims.(last.(full_gen)))
full_τ_est = vec(mean(full_τ_chn; dims = 1))
isapprox(full_τ_est, τ_obs; atol = .01)

partition_prior = BetaBinomialPartitionDistribution(k, 1, k)
eqs_mod = variances_equality_model(s, n, partition_prior)
eqs_spl = Simulations.get_sampler(eqs_mod)
eqs_chn = sample(eqs_mod, eqs_spl, 100_000)

eqs_gen = DynamicPPL.generated_quantities(eqs_mod, MCMCChains.get_sections(eqs_chn, :parameters))
eqs_τ_chn = reduce(vcat, permutedims.(last.(eqs_gen)))
eqs_τ_est = vec(mean(eqs_τ_chn; dims = 1))
isapprox(eqs_τ_est, τ_obs; atol = .1)

partition_samples = Int.(Array(group(eqs_chn, :partition)))
eqs_mat = compute_post_prob_eq(partition_samples)

eq_table = NA.NamedArray(
    Matrix(LA.UnitLowerTriangular(eqs_mat)),
    (data_reduced.abbreviation, data_reduced.abbreviation),
    ("Rows", "Cols")
)

function model_to_str(m, nms)
    str = first(nms)
    for i in 1:length(m) - 1
        str *= (m[i] == m[i + 1] ? " = " : " ≠ ") * nms[i+1]
    end
    return str
end
dict = compute_model_probs(partition_samples)
equation = model_to_str.(keys(dict), Ref(data_reduced.abbreviation))
postprob = collect(values(dict))
df_order_model = DF.DataFrame(
    equation     = equation,
    postprob   = postprob,
    priorprob  = pdf_model_distinct.(Ref(partition_prior), digits.(parse.(Int, keys(dict))))
)
df_order_prob = sort(df_order_model, :postprob, rev = true)

# postprob[end] / postprob[1] * (df_order_model.priorprob[end] / df_order_model.priorprob[1])



# df_order_model
df_order_prob
eq_table

fits = (
    full = 1 ./ full_τ_chn,
    eqs  = 1 ./ eqs_τ_chn
)

xlim = extrema([extrema(fits.full)...; extrema(fits.eqs)...])
xlim = (10, 25)
density_data = map(fits) do fit

    npoints = 2^12
    no_groups = size(fit, 2)

    x = Matrix{Float64}(undef, npoints, no_groups)
    y = Matrix{Float64}(undef, npoints, no_groups)

    for (i, col) in enumerate(eachcol(fit))
        k = KernelDensity.kde(col; npoints = npoints, boundary = xlim)
        x[:, i] .= k.x
        y[:, i] .= k.density
    end
    return (x = x, y = y)
end

nms = NamedTuple{keys(fits)}(keys(fits))

density_plots = map(nms) do key

    x, y = density_data[key]
    plt = plot(x, y;
        labels          = permutedims(data_reduced.abbreviation),
        legend_position = key === :full ? :topright : false,
        legendtitle     = "Population",#  : nothing,
        title           = key === :full ? "Full model" : "Model averaged",
        ylim            = key === :full ? (0.0, 0.8) : (0.0, 0.8),
        foreground_color_legend = nothing, background_color_legend = nothing
    )

end

# plot(randn(10, 3), randn(10, 3); labels = ["a" "b" "c"], legend_position = (0.1, 0.1))

density_plots.full
density_plots.eqs
plt_ylabel = plot([0 0]; ylab = "Density", showaxis = false, grid = false, axis = nothing, legend = false, left_margin = -6mm, right_margin = 8mm, ymirror=true)
left_panel = plot(
    plt_ylabel,
    density_plots.full,
    plot(density_plots.eqs, legend = false, xlab = "σ²"),
    bottom_margin = 3mm,
    layout = @layout [a{0.00001w} grid(2, 1)]
);


for i in 1:k, j in i:k
    eq_table[i, j] = NaN
end
x_nms = data_reduced.abbreviation
color_gradient = cgrad(cgrad(ColorSchemes.magma)[0.15:0.01:1.0])
annotations = []
for i in 1:k-1, j in i+1:k
    z = eq_table[k+1-i, k+1-j]
    col = color_gradient[1 - z]
    push!(annotations,
        (
            k - j + 0.5,
            i - 0.5,
            Plots.text(
                round_2_decimals(z),
                12, col, :center
            )
        )
    )
end

right_panel = heatmap(x_nms, reverse(x_nms), Matrix(eq_table)[k:-1:1, :],
    aspect_ratio = 1, showaxis = false, grid = false, color = color_gradient,
    clims = (0, 1),
    title = "Posterior probability of pairwise equality",
    #=colorbar_ticks = 0:.2:1, <- only works with pyplot =#
    annotate = annotations,
    xmirror = false);

joined_plot = plot(left_panel, right_panel, layout = (1, 2), size = (2, 1) .* 600);

savefig(joined_plot, "simulations/analysisLevenes/country_differences_BB_1_4.pdf")


#endregion