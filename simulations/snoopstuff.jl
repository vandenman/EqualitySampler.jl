using SnoopCompileCore
invalidations = @snoopr begin
    # package loads and/or method definitions that might invalidate other code
	using EqualitySampler
end
using SnoopCompile

length(uinvalidated(invalidations))

trees = invalidation_trees(invalidations)

ftrees = filtermod(EqualitySampler, trees)