using  Test
import StatsBase

@testset "Partition shortcuts" begin

	@testset "fast_countmap_partition" begin
		@testset "k = 2:7" begin
			for k in 2:7
				for model in EqualitySampler.DistinctModelsIterator(k)
					@test EqualitySampler.fast_countmap_partition(model) == collect(values(sort(StatsBase.countmap(model))))
				end
			end
		end
		@testset "random larger models" begin
			for k in BigInt(15):35
				model = rand(UniformMvUrnDistribution(k))
				@test EqualitySampler.fast_countmap_partition(model) == collect(values(sort(StatsBase.countmap(model))))
			end
		end
	end

	@testset "no_distinct_groups_in_partition" begin
		@testset "k = 2:7" begin
			for k in 2:7
				for model in EqualitySampler.DistinctModelsIterator(k)
					@test EqualitySampler.no_distinct_groups_in_partition(model) == length(Set(model))
				end
			end
		end
		@testset "random larger models" begin
			for k in BigInt(15):35
				model = rand(UniformMvUrnDistribution(k))
				@test EqualitySampler.no_distinct_groups_in_partition(model) == length(Set(model))
			end
		end
	end

end
