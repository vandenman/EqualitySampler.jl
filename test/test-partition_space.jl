@testset "PartitionSpace" begin

	@test_throws DomainError PartitionSpace(-1)
	@test_throws DomainError PartitionSpace(0)

	for k in 2:5
		@testset "  distinct partitions K = $k" begin

			it = PartitionSpace(k)
			@test length(it) == bellnum(k)
			@test Matrix(it) == EqualitySampler.generate_distinct_models(k)

			@inferred 					Tuple{Vector{Int64}, Vector{Int64}}  iterate(it)
			@inferred Union{Nothing, 	Tuple{Vector{Int64}, Vector{Int64}}} iterate(it, ones(Int, k))

		end
	end

	for k in 2:5
		@testset "duplicated partitions K = $k" begin

			it = PartitionSpace(k, EqualitySampler.DuplicatedPartitionSpace)
			@test length(it) == k^k
			@test all(x == collect(y) for (x, y) in zip(it, Iterators.product(fill(1:k, k)...)))

			@inferred 					Tuple{Vector{Int64}, Vector{Int64}}  iterate(it)
			@inferred Union{Nothing, 	Tuple{Vector{Int64}, Vector{Int64}}} iterate(it, ones(Int, k))

		end
	end
end
