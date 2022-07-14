using EqualitySampler, Plots, Statistics, NamedArrays, ProgressMeter, Plots.PlotMeasures, Chain
using Printf
import LogExpFunctions
include("priors_plot_colors_shapes_labels.jl")
include("simulation_helpers.jl")

# PairwiseMvUrnDistribution is not valid in general and only logpdf_model_distinct and pdf_model_distinct make sense
function _pairwiseMvUrnDistribution_helper(m, k, logp, logip)
	no_parameters = count_parameters(m)
	no_equal_pairs   = k - no_parameters + 1
	no_unequal_pairs = no_parameters - 1
	return no_equal_pairs * logp + no_unequal_pairs * logip
end

struct PairwiseMvUrnDistribution{T <: Integer} <: AbstractMvUrnDistribution{T}
	k::T
	p::Float64
	_logZ::Float64
	function PairwiseMvUrnDistribution(k::T, p::AbstractFloat = 0.5, logZ::AbstractFloat = 0.0) where T<:Integer
		new{T}(k, p, logZ)
	end
end
function PairwiseMvUrnDistribution(k::T, p::AbstractFloat = 0.5) where T<:Integer
	# logZ = LogExpFunctions.sum(
	# 	m->_pairwiseMvUrnDistribution_helper(m, k, log(p), log1p(-p)),
	# 	EqualitySampler.DistinctModelsIterator(k)
	# )
	logZ = LogExpFunctions.logsumexp(_pairwiseMvUrnDistribution_helper(m, k, log(p), log1p(-p)) for m in EqualitySampler.DistinctModelsIterator(k))
	PairwiseMvUrnDistribution(k, p, logZ)
end

EqualitySampler.logpdf_model_distinct(d::PairwiseMvUrnDistribution, x::AbstractVector{<:Integer}) = logpdf_model_distinct(d, count_parameters(x))
function EqualitySampler.logpdf_model_distinct(d::PairwiseMvUrnDistribution, no_parameters::Integer)
	EqualitySampler.in_eqsupport(d, no_parameters) || return -Inf
	no_equal_pairs   = d.k - no_parameters + 1
	no_unequal_pairs = no_parameters - 1
	no_equal_pairs * log(d.p) + no_unequal_pairs * log1p(-d.p) - d._logZ
end
function westfallMvUrnDistribution(k::T, pH0::AbstractFloat = 0.5, logZ::AbstractFloat = NaN) where T<:Integer
	τ = pH0 ^ (1 / k)
	# logτ = Float64((1 / k) * log(pH0))
	isnan(logZ) ? PairwiseMvUrnDistribution(k, Float64(τ)) : PairwiseMvUrnDistribution(k, Float64(τ), logZ)
end

# # pdf_model_distinct(westfallMvUrnDistribution(5, .5), [1, 1, 1, 1, 1]) ≈ .5
# @assert pdf_model_distinct(westfallMvUrnDistribution(5, .5, 0.0), [1, 1, 1, 1, 1]) ≈ .5
# @assert pdf_model_distinct(PairwiseMvUrnDistribution(5, .5, 0.0), [1, 1, 1, 1, 1]) ≈ .5^5
# @assert pdf_model_distinct(PairwiseMvUrnDistribution(5, .6, 0.0), [1, 1, 1, 2, 3]) ≈ .6^3 * .4^2

# mmm = Matrix(EqualitySampler.DistinctModelsIterator(5))
# d = westfallMvUrnDistribution(5, .5, 0.0)
# ppp = pdf_model_distinct.(Ref(d), eachcol(mmm))
# d2 = PairwiseMvUrnDistribution(5, 0.8705505632961241)
# ppp2 = pdf_model_distinct.(Ref(d2), eachcol(mmm))
# d3 = PairwiseMvUrnDistribution(5, 0.5)
# ppp3 = pdf_model_distinct.(Ref(d3), eachcol(mmm))
# [sum(ppp); sum(ppp2); sum(ppp3)]

# dp = PairwiseMvUrnDistribution(5, .5)
# dw = westfallMvUrnDistribution(5, .5)
# [pdf_model_distinct(d, [1, 1, 1, 1, 1]) for d in (dp, dw)]
# [pdf_model_distinct(d, [1, 1, 1, 1, 2]) for d in (dp, dw)]
# [pdf_model_distinct(d, [1, 1, 1, 2, 2]) for d in (dp, dw)]
# [pdf_model_distinct(d, [1, 1, 2, 2, 2]) for d in (dp, dw)]
# [pdf_model_distinct(d, [1, 1, 2, 2, 3]) for d in (dp, dw)]


# first(EqualitySampler.DistinctModelsIterator(3))
# collect(EqualitySampler.DistinctModelsIterator(3))
# Matrix(EqualitySampler.DistinctModelsIterator(3))

function compute_model_errors(true_ρ, ρ)
	α_error_count = 0
	β_error_count = 0
	α_errors_possible = 0
	β_errors_possible = 0
	for i in 1:length(ρ)-1, j in i+1:length(ρ)
		if true_ρ[i] == true_ρ[j]
			α_errors_possible += 1
			if ρ[i] != ρ[j]
				α_error_count += 1
			end
		elseif true_ρ[i] != true_ρ[j]
			β_errors_possible += 1
			if ρ[i] == ρ[j]
				β_error_count += 1
			end
		end
	end
	α_fam_error = α_error_count >= 1 ? 1 : 0
	α_prop_error = iszero(α_errors_possible) ? 0.0 : α_error_count / α_errors_possible
	β_prop_error = iszero(β_errors_possible) ? 0.0 : β_error_count / β_errors_possible
	return (; α_fam_error, α_prop_error, β_prop_error, α_errors_possible, β_errors_possible)
end


# function barrier
function get_prob_ρ(s, ρ, z_PairwiseMvUrnDistribution, z_westfallMvUrnDistribution)
	if s !== :Westfall && s !== :Westfall_uncorrected
		d = instantiate_prior(s, k)
		prob_ρ = pdf_model_distinct(d, ρ)
	elseif s === :Westfall_uncorrected
		d = PairwiseMvUrnDistribution(k, .5, z_PairwiseMvUrnDistribution)
		prob_ρ = pdf_model_distinct(d, ρ)
		# prob_ρ = pdf_model_distinct(UniformMvUrnDistribution(k), ρ)
	else #s !== :Westfall
		d = westfallMvUrnDistribution(k, .5, z_westfallMvUrnDistribution)
		prob_ρ = pdf_model_distinct(d, ρ)
		# prob_ρ = pdf_model_distinct(UniformMvUrnDistribution(k), ρ)
	end
	return prob_ρ::Float64
end

function compute_prior_performance(k::Integer, priors_sym, hypotheses)
	α_fam_errors = zeros(length(priors_sym), length(hypotheses))
	α_errors = zeros(length(priors_sym), length(hypotheses))
	β_errors = zeros(length(priors_sym), length(hypotheses))

	z_PairwiseMvUrnDistribution = PairwiseMvUrnDistribution(k, .5)._logZ
	τ = 0.5 ^ (1 / k)
	z_westfallMvUrnDistribution = LogExpFunctions.logsumexp(_pairwiseMvUrnDistribution_helper(m, k, log(τ),  log1p(-τ))  for m in EqualitySampler.DistinctModelsIterator(k))

	hypo_counts = zeros(Int, k)
	@showprogress for true_ρ in EqualitySampler.DistinctModelsIterator(k)
		j = EqualitySampler.count_parameters(true_ρ)
		for ρ in EqualitySampler.DistinctModelsIterator(k)

			α_fam_error, α_prop_error, β_prop_error = compute_model_errors(true_ρ, ρ)
			hypo_counts[j] += 1

			for (l, s) in enumerate(priors_sym)

				prob_ρ = get_prob_ρ(s, ρ, z_PairwiseMvUrnDistribution, z_westfallMvUrnDistribution)
				α_fam_errors[l, j] += α_fam_error  * prob_ρ
				α_errors[l, j]     += α_prop_error * prob_ρ
				β_errors[l, j]     += β_prop_error * prob_ρ


			end
		end
	end
	return α_fam_errors, α_errors, β_errors, hypo_counts
end

# @code_warntype compute_prior_performance(k, nsim, priors_sym, hypotheses)

k = 5
priors_sym = get_priors()
priors_sym = priors_sym[[1:2; 4:7; 9:11]] # drop :BetaBinomialk1 and :DirichletProcess2_0
priors_sym = priors_sym[[2:9; 1]] # reorder
hypotheses = (:p00, :p25, :p50, :p75, :p100)

α_fam_errors, α_errors, β_errors, hypo_counts = compute_prior_performance(k, priors_sym, hypotheses)
hypo_counts = stirlings2.(k, 1:k)
for i in axes(α_fam_errors, 1)
	α_fam_errors[i, :] = α_fam_errors[i, :] ./ hypo_counts
	α_errors[i, :]     = α_errors[i, :]     ./ hypo_counts
	β_errors[i, :]     = β_errors[i, :]     ./ hypo_counts
end

formatter(x) = @sprintf("%.7f", x)
to_pretty_table(x) = NamedArray(x, (collect(string.(priors_sym)), collect(hypothesis_to_inequalities.(hypotheses, k))), ("Prior", "Inequalities"))
to_pretty_table2(x) = NamedArray(formatter.(hcat(x, mean(view(x, :, 1:k-1); dims = 2))), (collect(string.(priors_sym)), [string.(collect(hypothesis_to_inequalities.(hypotheses, k))); "average"]), ("Prior", "Inequalities"))


to_pretty_table(α_fam_errors)
to_pretty_table(α_errors)
to_pretty_table(β_errors)

to_pretty_table2(α_fam_errors)
to_pretty_table2(α_errors)
to_pretty_table2(β_errors)

[string.(collect(hypothesis_to_inequalities.(hypotheses, k))); "average"]

α_fam_errors2 = hcat(α_fam_errors, mean(view(α_fam_errors, :, 1:k-1); dims = 2))
α_errors2     = hcat(α_errors,     mean(view(α_errors,     :, 1:k-1); dims = 2))
β_errors2     = hcat(β_errors,     mean(view(β_errors,     :, 2:k);   dims = 2))

xlabels = (1:6, [string.(collect(hypothesis_to_inequalities.(hypotheses, k))); "average"])
xvalues = Float64.(reshape(repeat(1:length(hypotheses)+1, inner = size(α_fam_errors2, 1)), size(α_fam_errors2)...))
jitterwidth = 0.25
jittervalues = range(-jitterwidth, jitterwidth, length = size(xvalues, 1))
jittervalues = jittervalues .- mean(jittervalues)
for i in axes(xvalues, 2)
	xvalues[:, i] .+= jittervalues
end

labels = get_labels(priors_sym)
labels[2:3] .= labels[3:-1:2]
colors = reshape(repeat(get_colors(priors_sym), size(α_fam_errors2, 2)), size(α_fam_errors2)...)
shapes = reshape(repeat(get_shapes(priors_sym), size(α_fam_errors2, 2)), size(α_fam_errors2)...)

function do_plot(errors, ylab; kwargs...)
	plot(xvalues', errors',
	labels = labels, color = colors', shape = permutedims(shapes),
	foreground_color_legend = nothing, background_color_legend = nothing,
	ylab = ylab, xlab = "No. inequalities", xticks = xlabels; kwargs...)
end
plt_α_fam = do_plot(α_fam_errors2, "Familywise α error"; markersize = 8, #=yticks = 0:5:25, ylim = (0, 25),=# legend = (.7, 1))
plt_α_err = do_plot(α_errors2,     "α error";  markersize = 8,#=yticks = 0:5:20, ylim = (0, 20),=# legend = false)
plt_β_err = do_plot(β_errors2,     "β error",  markersize = 8,legend = false)

plt_joined = plot(plt_α_fam, plt_α_err, plt_β_err, layout = (1, 3), size = (3, 1).* 800,
	bottom_margin = 14mm, left_margin = 16mm)
savefig(plt_joined, joinpath("figures", "prior_performance.pdf"))


k = 3
true_ρ = ones(Int, k)
true_ρ = collect(1:k)
true_ρ = ones(Int, k)

α_fam_error_prop, α_prop_error_prop, β_prop_error_prop = zeros(k), zeros(k), zeros(k)
for true_ρ in EqualitySampler.DistinctModelsIterator(k)
	j = EqualitySampler.count_parameters(true_ρ)
	for ρ in EqualitySampler.DistinctModelsIterator(k)
		α_fam_error, α_prop_error, β_prop_error = compute_model_errors(true_ρ, ρ)
		α_fam_error_prop[j] += α_fam_error
		α_prop_error_prop[j] += α_prop_error
		β_prop_error_prop[j] += β_prop_error
	end
end
α_fam_error_prop  ./ stirlings2.(k, 1:k) # the number of models that make an alpha error

counts, sizes = count_set_partitions_given_partition_size(k)
α_prop_error_prop ./ stirlings2.(k, 1:k)
β_prop_error_prop ./ stirlings2.(k, 1:k)
sizes
mmms = Matrix{Int}(undef, k, length(sizes))
idx = 1
for v in sizes
	mmms[:, idx] .= reduce(vcat, fill(i, c) for (i, c) in enumerate(v))
	idx += 1
end

using DataFrames
k = 5

function compute_sum_partition_eqs(k)
	res = 0
	for ρ in eachcol(generate_distinct_models(k)), i in 1:k-1, j in i+1:k
		res += ρ[i] == ρ[j] ? 1 : 0
	end
	res
end
[compute_sum_partition_eqs(i) for i in 2:8] # https://oeis.org/A105488

compute_sum_partition_eqs_oeis(k) = binomial(k, 2) * bellnumr(k-1, 0)
[compute_sum_partition_eqs_oeis(i) for i in 2:8]

function compute_hhh(k)
	nnn = length(EqualitySampler.DistinctModelsIterator(k))^2
	hhh = DataFrame(
		:ρ_true => Vector{String}(undef, nnn),
		:ρ      => Vector{String}(undef, nnn),
		:α_c    => Vector{Int}(undef, nnn),
		:α_t    => Vector{Int}(undef, nnn),
		:β_c    => Vector{Int}(undef, nnn),
		:β_t    => Vector{Int}(undef, nnn),
	)
	# mmm = collect(EqualitySampler.DistinctModelsIterator(k))
	mmm = collect(eachcol(generate_distinct_models(k)))
	idx = 1
	for m in mmm
		tmp = compute_model_errors.(Ref(m), mmm)
		for j in eachindex(tmp)
			hhh[idx, :ρ_true] = join(m)
			hhh[idx, :ρ]      = join(mmm[j])
			hhh[idx, :α_c]    = round(Int, tmp[j].α_prop_error * tmp[j].α_errors_possible)
			hhh[idx, :α_t]    = tmp[j].α_errors_possible
			hhh[idx, :β_c]    = round(Int, tmp[j].β_prop_error * tmp[j].β_errors_possible)
			hhh[idx, :β_t]    = tmp[j].β_errors_possible
			idx += 1
		end
	end
	return hhh
end
hhh = compute_hhh(5)
show(hhh, allrows=true)
sum(hhh.α_c)
sum(hhh.α_t)
sum(ifelse.(iszero.(hhh.α_c), 0.0, hhh.α_c ./ hhh.α_t))
sum(ifelse.(iszero.(hhh.β_c), 0.0, hhh.β_c ./ hhh.β_t))

sum(hhh.β_c)
sum(hhh.β_t)


hhh = compute_hhh(3)
function foo(hhh)
	@chain hhh begin
		transform(:ρ_true => (x->count_equalities.(x)) => :no_equalities)
		groupby(:no_equalities)
		combine(:α_c=>sum, :β_c=>sum, :α_t=>sum, :β_t=>sum, :no_equalities=>first)
		# combine(:α_c=>sum, :β_c=>sum, :α_t=>first, :β_t=>first, :no_equalities=>first)
	end
end
[foo(compute_hhh(k)).α_c_sum ./ bellnum(k) for k in 2:5]

[sum(compute_hhh(k).α_c) for k in 2:6]
[sum(compute_hhh(k).α_t) for k in 2:6] # https://oeis.org/A105488 * B(k)
[sum(compute_hhh(k).β_c) for k in 2:6]
[sum(compute_hhh(k).β_t) for k in 2:6]
# α_c and β_c follow https://oeis.org/A193317
fast_sum_α_c(k) = binomial(k, 2) * bellnum(k-1) * (bellnum(k) - bellnum(k-1))
fast_sum_α_c.(2:6)

ks = 2:8
vs = zeros(Int, length(ks))
for i in eachindex(ks)
	hhh = compute_hhh(ks[i])
	vs[i] = hhh.α_t[1] + hhh.β_t[1]
end
# https://oeis.org/A000217
binomial.(ks, 2)



ks = 2:7
vs = zeros(Int, length(ks))
for i in eachindex(ks)
	hhh = compute_hhh(ks[i])
	vs[i] = hhh.α_t[1]# + hhh.β_t[1]
end

function compute_sum_partition2(k)
	# res = zeros(Int, k, k)
	res = zeros(Int, binomial(k, 2))
	mmm = eachcol(generate_distinct_models(k))
	for m in mmm
		idx = 1
		for i in 1:k-1, j in i+1:k
			if m[i] != m[j]
				# res[i, j] += 1
				res[idx] += 1
			end
			idx += 1
		end
	end
	return res
end
rrr=[compute_sum_partition2(i) for i in 2:8] # https://oeis.org/A005493
first.(rrr)
show(compute_hhh(3), allrows=true)

compute_sum_partition2_fast(k) = sum(i * stirlings2(k-1, i) for i in 1:k-1)

compute_sum_partition2_fast(4)
hhh = compute_hhh(4)
@chain hhh begin
	subset(:ρ_true => (x->x .==("1111")))
	combine(:α_c=>sum)
end

[compute_sum_partition2_fast(i) for i in 2:8]

function compute_sum_partition3(k)
	# res = zeros(Int, k, k)
	res = zeros(Int, binomial(k, 2))
	mmm = eachcol(generate_distinct_models(k))
	for m in mmm
		idx = 1
		for i in 1:k-1, j in i+1:k
			if m[i] == m[j]
				# res[i, j] += 1
				res[idx] += 1
			end
			idx += 1
		end
	end
	return res
end
rrr=[compute_sum_partition3(i) for i in 2:8] # https://oeis.org/A000110
first.(rrr)
bellnumr.(1:7, 0)

function goo(k)
	res = 0.0
	for p in EqualitySampler.DistinctModelsIterator(k)
		for q in EqualitySampler.DistinctModelsIterator(k)

			num = 0
			den = 0

			for i in 1:k-1, j in i+1:k
				num += (p[i] == p[j]) * (q[i] != q[j])
				den += (p[i] == p[j])
			end
			if !iszero(den)
				res += num / den
			end
		end
	end
	return res
end
function hoo(k)
	# res = 0.0
	length(EqualitySampler.DistinctModelsIterator(k)) * compute_sum_partition2_fast(k)

	# for p in EqualitySampler.DistinctModelsIterator(k)
	# 	for q in EqualitySampler.DistinctModelsIterator(k)

	# 		num = 0
	# 		den = 0

	# 		for i in 1:k-1, j in i+1:k
	# 			num += (p[i] == p[j]) * (q[i] != q[j])
	# 			den += (p[i] == p[j])
	# 		end
	# 		if !iszero(den)
	# 			res += num / den
	# 		end
	# 	end
	# end
	# return res
end
goo(2)
hoo(2)
goo.(2:5)
(length.(EqualitySampler.DistinctModelsIterator.(2:5)) .- 1) .* compute_sum_partition2_fast.(2:5)

function loo(p)
	k = length(p)
	res = 0.0
	for q in EqualitySampler.DistinctModelsIterator(k)

		num = 0
		den = 0

		for i in 1:k-1, j in i+1:k
			num += (p[i] == p[j]) * (q[i] != q[j])
			den += (p[i] == p[j])
		end
		if !iszero(den)
			res += num / den
		end
	end
	return res
end
for p in EqualitySampler.DistinctModelsIterator(3)
	println(p)
end
[println(p) for p in EqualitySampler.DistinctModelsIterator(3)]
[loo(p) for p in collect(EqualitySampler.DistinctModelsIterator(3))]
[loo(p) for p in collect(EqualitySampler.DistinctModelsIterator(3))]

loo([1, 1, 1])
loo([1, 1, 2])
loo([1, 2, 1])
loo([1, 2, 2])
loo([1, 1, 1])

ρ0 = [1, 1, 1]
ρ1 = [1, 1, 2]
[abs(a - b) for a in ρ0, b in ρ1]
ρ1 * ρ1'
ρ0 * ρ1'
ρ0 * ρ1'

sum(hhh.α_t)
sum(hhh.β_c)
show(hhh, allrows=true, allcols=true)
sum(hhh.β_c)

# NOTE: possible errors

# shortcut!
sum(getindex.(compute_model_errors.(Ref(ones(Int, k)), eachcol(mmms)), 2) .* counts)
sum(getindex.(compute_model_errors.(Ref(ones(Int, k)), eachcol(mmms)), 2) .* counts)
sum(getindex.(compute_model_errors.(Ref(collect(1:k)), eachcol(mmms)), 3) .* counts)
# need all instances because mmms only contains 1
mean([
	sum(getindex.(compute_model_errors.(Ref([1, 1, 2]), eachcol(mmms)), 2) .* counts),
	sum(getindex.(compute_model_errors.(Ref([1, 2, 1]), eachcol(mmms)), 2) .* counts),
	sum(getindex.(compute_model_errors.(Ref([1, 2, 2]), eachcol(mmms)), 2) .* counts)
])

mean([
	sum(getindex.(compute_model_errors.(Ref([1, 1, 2]), eachcol(mmms)), 3) .* counts),
	sum(getindex.(compute_model_errors.(Ref([1, 2, 1]), eachcol(mmms)), 3) .* counts),
	sum(getindex.(compute_model_errors.(Ref([2, 1, 1]), eachcol(mmms)), 3) .* counts)
])


# mean([
# 	sum(getindex.(compute_model_errors.(Ref([1, 1, 1, 1, 2]), eachcol(mmms)), 2) .* counts),
# 	sum(getindex.(compute_model_errors.(Ref([1, 1, 1, 2, 1]), eachcol(mmms)), 2) .* counts),
# 	sum(getindex.(compute_model_errors.(Ref([1, 1, 2, 1, 1]), eachcol(mmms)), 2) .* counts),
# 	sum(getindex.(compute_model_errors.(Ref([1, 2, 1, 1, 1]), eachcol(mmms)), 2) .* counts),
# 	sum(getindex.(compute_model_errors.(Ref([2, 1, 1, 1, 1]), eachcol(mmms)), 2) .* counts)
# ])

# TODO: maybe work this out for k=3 or 4 where the space is easier to enumerate...

mm = [fill(i, c) for v in sizes for (i, c) in enumerate(v)]

β_prop_error_prop ./ stirlings2.(k, 1:k)

# (α_fam_error_prop, α_prop_error_prop, β_prop_error_prop)
(α_prop_error_prop, β_prop_error_prop)
α_prop_error_prop .+ reverse(β_prop_error_prop)




# α familywise error
# Prior ╲ Inequalities │           0            1            2            3            4      average
# ─────────────────────┼─────────────────────────────────────────────────────────────────────────────
# uniform              │ "0.0009808"  "0.0144231"  "0.0225962"  "0.0071154"  "0.0000000"  "0.0112788"
# BetaBinomial11       │ "0.0008000"  "0.0118000"  "0.0188000"  "0.0063867"  "0.0000000"  "0.0094467"
# BetaBinomialk1       │ "0.0009921"  "0.0148413"  "0.0244841"  "0.0091720"  "0.0000000"  "0.0123724"
# BetaBinomial1k       │ "0.0004444"  "0.0063889"  "0.0096032"  "0.0028228"  "0.0000000"  "0.0048148"
# BetaBinomial1binomk2 │ "0.0002857"  "0.0040659"  "0.0059890"  "0.0016896"  "0.0000000"  "0.0030076"
# DirichletProcess0_5  │ "0.0005937"  "0.0084815"  "0.0122381"  "0.0033333"  "0.0000000"  "0.0061616"
# DirichletProcess1_0  │ "0.0008000"  "0.0115833"  "0.0172917"  "0.0050000"  "0.0000000"  "0.0086688"
# DirichletProcess2_0  │ "0.0009333"  "0.0137222"  "0.0213333"  "0.0066667"  "0.0000000"  "0.0106639"
# DirichletProcessGP   │ "0.0009453"  "0.0139268"  "0.0217774"  "0.0068880"  "0.0000000"  "0.0108844"
# Westfall             │ "0.0009808"  "0.0144231"  "0.0225962"  "0.0071154"  "0.0000000"  "0.0112788"
# Westfall_uncorrected │ "0.0009808"  "0.0144231"  "0.0225962"  "0.0071154"  "0.0000000"  "0.0112788"

# α error
# Prior ╲ Inequalities │           0            1            2            3            4      average
# ─────────────────────┼─────────────────────────────────────────────────────────────────────────────
# uniform              │ "0.0007115"  "0.0106731"  "0.0177885"  "0.0071154"  "0.0000000"  "0.0090721"
# BetaBinomial11       │ "0.0006387"  "0.0095800"  "0.0159667"  "0.0063867"  "0.0000000"  "0.0081430"
# BetaBinomialk1       │ "0.0009172"  "0.0137579"  "0.0229299"  "0.0091720"  "0.0000000"  "0.0116942"
# BetaBinomial1k       │ "0.0002823"  "0.0042341"  "0.0070569"  "0.0028228"  "0.0000000"  "0.0035990"
# BetaBinomial1binomk2 │ "0.0001690"  "0.0025345"  "0.0042241"  "0.0016896"  "0.0000000"  "0.0021543"
# DirichletProcess0_5  │ "0.0003333"  "0.0050000"  "0.0083333"  "0.0033333"  "0.0000000"  "0.0042500"
# DirichletProcess1_0  │ "0.0005000"  "0.0075000"  "0.0125000"  "0.0050000"  "0.0000000"  "0.0063750"
# DirichletProcess2_0  │ "0.0006667"  "0.0100000"  "0.0166667"  "0.0066667"  "0.0000000"  "0.0085000"
# DirichletProcessGP   │ "0.0006888"  "0.0103320"  "0.0172200"  "0.0068880"  "0.0000000"  "0.0087822"
# Westfall             │ "0.0007115"  "0.0106731"  "0.0177885"  "0.0071154"  "0.0000000"  "0.0090721"
# Westfall_uncorrected │ "0.0007115"  "0.0106731"  "0.0177885"  "0.0071154"  "0.0000000"  "0.0090721"

# β error
# Prior ╲ Inequalities │           0            1            2            3            4      average
# ─────────────────────┼─────────────────────────────────────────────────────────────────────────────
# uniform              │ "0.0000000"  "0.0043269"  "0.0072115"  "0.0028846"  "0.0002885"  "0.0036058"
# BetaBinomial11       │ "0.0000000"  "0.0054200"  "0.0090333"  "0.0036133"  "0.0003613"  "0.0045167"
# BetaBinomialk1       │ "0.0000000"  "0.0012421"  "0.0020701"  "0.0008280"  "0.0000828"  "0.0010351"
# BetaBinomial1k       │ "0.0000000"  "0.0107659"  "0.0179431"  "0.0071772"  "0.0007177"  "0.0089716"
# BetaBinomial1binomk2 │ "0.0000000"  "0.0124655"  "0.0207759"  "0.0083104"  "0.0008310"  "0.0103879"
# DirichletProcess0_5  │ "0.0000000"  "0.0100000"  "0.0166667"  "0.0066667"  "0.0006667"  "0.0083333"
# DirichletProcess1_0  │ "0.0000000"  "0.0075000"  "0.0125000"  "0.0050000"  "0.0005000"  "0.0062500"
# DirichletProcess2_0  │ "0.0000000"  "0.0050000"  "0.0083333"  "0.0033333"  "0.0003333"  "0.0041667"
# DirichletProcessGP   │ "0.0000000"  "0.0046680"  "0.0077800"  "0.0031120"  "0.0003112"  "0.0038900"
# Westfall             │ "0.0000000"  "0.0043269"  "0.0072115"  "0.0028846"  "0.0002885"  "0.0036058"
# Westfall_uncorrected │ "0.0000000"  "0.0043269"  "0.0072115"  "0.0028846"  "0.0002885"  "0.0036058"


# for true_ρ in all_possible_partitions_of_size(5)

# 	j = count_inequalities(true_ρ) + 1

# 	for ρ in all_possible_partitions_of_size(5)

# 		α_fam_error, α_prop_error, β_prop_error = compute_model_errors(true_ρ, ρ)

# 		for (l, prior) in enumerate(priors)

# 			prob_ρ = get_prob_ρ(prior, ρ)
# 			α_fam_errors[l, j] += α_fam_error  * prob_ρ
# 			α_errors[l, j]     += α_prop_error * prob_ρ
# 			β_errors[l, j]     += β_prop_error * prob_ρ

# 		end
# 	end
# end