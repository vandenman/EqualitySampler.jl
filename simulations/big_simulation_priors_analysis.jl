using EqualitySampler, Plots, Statistics, NamedArrays, ProgressMeter
using Printf
include("priors_plot_colors_shapes_labels.jl")
include("simulation_helpers.jl")

function compute_model_errors(true_ρ, ρ)
	α_error_count = 0
	β_error_count = 0
	α_errors_possible = 0
	β_errors_possible = 0
	for i in 1:length(ρ)-1, j in i+1:length(ρ)
		if true_ρ[i] == true_ρ[j]
			α_errors_possible += 1
			if ρ[i] != ρ[j]
				# @show ρ[i], ρ[j], i, j
				α_error_count += 1
			end
		elseif true_ρ[i] != true_ρ[j]
			β_errors_possible += 1
			if ρ[i] == ρ[j]
				β_error_count +=1
			end
		end
	end
	α_fam_error = α_error_count >= 1 ? 1 : 0
	α_prop_error = iszero(α_errors_possible) ? 0.0 : α_error_count / α_errors_possible
	β_prop_error = iszero(β_errors_possible) ? 0.0 : β_error_count / β_errors_possible
	return (; α_fam_error, α_prop_error, β_prop_error)
end

# function barrier
function get_prob_ρ(s, ρ)
	if s !== :Westfall && s !== :Westfall_uncorrected
		d = instantiate_prior(s, k)
		prob_ρ = pdf_model_distinct(d, ρ)
	elseif s !== :Westfall_uncorrected
		prob_ρ = pdf_model_distinct(UniformMvUrnDistribution(k), ρ)
	else #s !== :Westfall
		prob_ρ = pdf_model_distinct(UniformMvUrnDistribution(k), ρ)
	end
	return prob_ρ::Float64
end

function compute_prior_performance(k::Integer, priors_sym, hypotheses)
	α_fam_errors = zeros(length(priors_sym), length(hypotheses))
	α_errors = zeros(length(priors_sym), length(hypotheses))
	β_errors = zeros(length(priors_sym), length(hypotheses))

	@showprogress for true_ρ in EqualitySampler.DistinctModelsIterator(k)
		j = EqualitySampler.count_parameters(true_ρ)
		for ρ in EqualitySampler.DistinctModelsIterator(k)

			α_fam_error, α_prop_error, β_prop_error = compute_model_errors(true_ρ, ρ)

			for (l, s) in enumerate(priors_sym)

				prob_ρ = get_prob_ρ(s, ρ)
				α_fam_errors[l, j] += α_fam_error  * prob_ρ
				α_errors[l, j]     += α_prop_error * prob_ρ
				β_errors[l, j]     += β_prop_error * prob_ρ

			end
		end
	end
	return α_fam_errors, α_errors, β_errors
end

# @code_warntype compute_prior_performance(k, nsim, priors_sym, hypotheses)

k = 5
nsim = 1_000
priors_sym = get_priors()
hypotheses = (:p00, :p25, :p50, :p75, :p100)

α_fam_errors, α_errors, β_errors = compute_prior_performance(k, priors_sym, hypotheses)

α_fam_errors_avg = dropdims(mean(α_fam_errors; dims = 3); dims = 3)
α_errors_avg     = dropdims(mean(α_errors;     dims = 3); dims = 3)
β_errors_avg     = dropdims(mean(β_errors;     dims = 3); dims = 3)

formatter(x) = @sprintf("%.7f", x)
to_pretty_table(x) = NamedArray(x, (collect(string.(priors_sym)), collect(hypothesis_to_inequalities.(hypotheses, k))), ("Prior", "Inequalities"))
to_pretty_table2(x) = NamedArray(formatter.(hcat(x, mean(view(x, :, 1:k-1); dims = 2))), (collect(string.(priors_sym)), [string.(collect(hypothesis_to_inequalities.(hypotheses, k))); "average"]), ("Prior", "Inequalities"))


to_pretty_table(α_fam_errors)
to_pretty_table(α_errors)
to_pretty_table(β_errors)

to_pretty_table2(α_fam_errors)
to_pretty_table2(α_errors)
to_pretty_table2(β_errors)




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