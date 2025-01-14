using  Test
import StatsBase, OrderedCollections

@testset "Partition shortcuts" begin

    reference(x) = collect(values(sort!(OrderedCollections.OrderedDict(StatsBase.countmap(x)))))

    @testset "fast_countmap_partition" begin
        @testset "k = 2:7" begin
            for k in 2:7
                for model in PartitionSpace(k)
                    @test EqualitySampler.fast_countmap_partition(model) == reference(model)
                end
            end
        end
        @testset "random larger models" begin
            for k in BigInt(15):35
                model = rand(UniformPartitionDistribution(k))
                @test EqualitySampler.fast_countmap_partition(model) == reference(model)
            end
        end
    end

    @testset "no_distinct_groups_in_partition" begin
        @testset "k = 2:7" begin
            for k in 2:7
                for model in PartitionSpace(k)
                    @test EqualitySampler.no_distinct_groups_in_partition(model) == length(Set(model))
                end
            end
        end
        @testset "random larger models" begin
            for k in BigInt(15):35
                model = rand(UniformPartitionDistribution(k))
                check = EqualitySampler.no_distinct_groups_in_partition(model) == length(Set(model))
                !check && @show model
                @test EqualitySampler.no_distinct_groups_in_partition(model) == length(Set(model))
            end
        end
    end

    @testset "fast contains" begin
        @testset "k = 2:5 duplicated" begin
            for k in 2:5
                for model in PartitionSpace(k, EqualitySampler.DuplicatedPartitionSpace)
                    s = Set(model)
                    partition_number = EqualitySampler.partition_to_number(model)
                    for i in eachindex(model)
                        @test (i ∈ s) == EqualitySampler.label_in_partition(i, partition_number)
                    end
                end
            end
        end
        @testset "k = 2:6 distinct" begin
            for k in 2:6
                for model in PartitionSpace(k)
                    s = Set(model)
                    partition_number = EqualitySampler.partition_to_number(model)
                    for i in eachindex(model)
                        for i in eachindex(model)
                            @test (i ∈ s) == EqualitySampler.label_in_partition(i, partition_number)
                        end
                    end
                end
            end
        end
    end
end
