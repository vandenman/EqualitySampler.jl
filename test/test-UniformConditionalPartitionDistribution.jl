using Test
println(pwd())
if endswith(pwd(), "bfvartest")
	cd("julia")
elseif endswith(pwd(), "test")
	cd("../")
end
println(pwd())
include(joinpath(pwd(), "test", "helper-UniformConditionalPartitionDistribution.jl"))
include(joinpath(pwd(), "UniformConditionalPartitionDistribution.jl"))

max_models = 8
enumerate_models.(1:max_models); # precompute models

function compare_brute_force_to_new(n::Int)
	ns = n:max_models
	patterns = visited_models[n - 1]
	count_bell			= get_conditional_counts.(ns, Ref(patterns))
	count_brute_force 	= list_counts.(ns, Ref(patterns))
	return count_bell, count_brute_force
end

flatten(x) = permutedims(hcat(x...))

@testset "compare brute force to manual" begin
	for i in 2:max_models
		# println("Comparing $i")
		a1, a2 = compare_brute_force_to_new(i)
		a3 = flatten.(collect.(values.(a2)))
		@test a1 == a3
	end
end

@testset "test uniformity of sampling" begin

	startvecs = [collect(1:n) for n in 2:max_models]

	for i in eachindex(startvecs)
		# println("Comparing $i")
		p_model, p_equal = count_probs_and_models(startvecs[i], 100_000)

		prob_models = collect(values(p_model)) ./ 100_000
		expected_prob = 1 / length(p_model)
		diff = abs.(prob_models .- expected_prob)
		max_abs_tol = max(0.001, 10^floor(log10(expected_prob)))
		@test all(<(max_abs_tol), diff)

		p_equal_vec = collect(p_equal[j, i] for i in 1:size(p_equal)[1]-1 for j in i+1:size(p_equal)[2])
		expected_eq_prob = round(p_equal_vec[1], digits = 2)
		@test all(x-> abs(x - expected_eq_prob) <.01, p_equal_vec)
	end

end
