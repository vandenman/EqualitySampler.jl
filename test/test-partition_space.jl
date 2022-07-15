@testset "partition_space" begin

    for k in 2:4

        it = partition_space(k)
        @test length(it) == bellnum(k)
        @test Matrix(it) == EqualitySampler.generate_distinct_models(k)
        @inferred Union{Nothing, Tuple{Vector{Int64}, Int64}} iterate(partition_space(k))

    end

end