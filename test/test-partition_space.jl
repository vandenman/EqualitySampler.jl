@testset "PartitionSpace" begin

    """
    generate_distinct_models(k::Int)

    Generates all distinct models that represent equalities.
    Deprecated in favor of `partition_space(k::Int)`.
    """
    function generate_distinct_models(k::Int)
        # based on https://stackoverflow.com/a/30898130/4917834
        current = ones(Int, k)
        no_models = bellnum(k)
        result = Matrix{Int}(undef, k, no_models)
        result[:, 1] .= current
        isone(k) && return result
        range = k:-1:2
        for i in 2:no_models

            idx = findfirst(i->current[i] < k && any(==(current[i]), current[1:i-1]), range)
            rightmost_incrementable = range[idx]
            current[rightmost_incrementable] += 1
            current[rightmost_incrementable + 1 : end] .= 1
            result[:, i] .= current

        end
        return result
    end

    @test_throws DomainError PartitionSpace(-1)
    @test_throws DomainError PartitionSpace(0)

    for k in 2:5
        @testset "  distinct partitions K = $k" begin

            it = PartitionSpace(k)
            @test length(it) == bellnum(k)
            @test Matrix(it) == generate_distinct_models(k)

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

    @testset "eltype and collect infer" begin
        for T in (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Int128, UInt128)
            @test eltype(PartitionSpace(T(3))) == Vector{T}
            @inferred Vector{Vector{T}} collect(PartitionSpace(3))
        end
    end
end
