@testset "partition_space" begin

	@test_throws DomainError partition_space(-1)
	@test_throws DomainError partition_space(0)

	for k in 2:5
		@testset "distinct partitions K = $k" begin

			it = partition_space(k)
			@test length(it) == bellnum(k)
			@test Matrix(it) == EqualitySampler.generate_distinct_models(k)

			@inferred 					Tuple{Vector{Int64}, Vector{Int64}}  iterate(it)
			@inferred Union{Nothing, 	Tuple{Vector{Int64}, Vector{Int64}}} iterate(it, ones(Int, k))

		end
	end

	for k in 2:5
		@testset "duplicated partitions K = $k" begin

			it = partition_space(k, EqualitySampler.DuplicatedPartitionSpace)
			@test length(it) == k^k
			@test all(x == collect(y) for (x, y) in zip(it, Iterators.product(fill(1:k, k)...)))

			@inferred 					Tuple{Vector{Int64}, Vector{Int64}}  iterate(it)
			@inferred Union{Nothing, 	Tuple{Vector{Int64}, Vector{Int64}}} iterate(it, ones(Int, k))

		end
	end
end
