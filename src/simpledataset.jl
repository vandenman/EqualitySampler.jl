struct SimpleDataSet
	y::Vector{Float64}
	g::Vector{UnitRange{Int}}
	function SimpleDataSet(y::Vector{Float64}, g::Vector{UnitRange{Int}})
		@assert length(y) == sum(length, g)
		for i in 1:length(g)-1
			for j in i+1:length(g)
				@assert isempty(intersect(g[i], g[j]))
			end
		end
		return new(y, g)
	end
end

function SimpleDataSet(y::Vector{Float64}, g::Vector{<:Integer})

	@assert length(y) == length(g)
	o = sortperm(g)
	g_sorted = g[o]
	u = unique(g_sorted)

	g_unitrange = Vector{UnitRange{Int}}(undef, length(u))
	g_from = 1
	g_to = 1
	idx = 1
	for i in eachindex(g)
		if g_sorted[i] == u[idx]
			g_to += 1
		else
			g_unitrange[idx] = g_from:g_to
			idx += 1
			g_from = g_to + 1
		end
	end
	g_unitrange[idx] = g_from:length(g)

	SimpleDataSet(y[o], g_unitrange)

end

function SimpleDataSet(f::StatsModels.FormulaTerm, df::DataFrames.DataFrame)
	ts = StatsModels.apply_schema(f, StatsModels.schema(df))
	ts = StatsModels.drop_term(ts, StatsModels.term(1))

	!isone(length(ts.rhs.terms)) && throw(DomainError(f, "Expected `$f` to only specify one predictor, for example `y ~ g`"))
	y, g = map(vec, StatsModels.modelcols(ts, df))

	return SimpleDataSet(y, g)
end

function Base.show(io::IO, ::MIME"text/plain", x::SimpleDataSet)
	println(io, "SimpleDataSet of $(length(x.g)) groups:")
	println(IOContext(io, :compact=>true, :limit=>true), "            data: ", x.y)
	println(IOContext(io, :compact=>true, :limit=>true), "group membership: ", x.g)
	println(IOContext(io, :compact=>true, :limit=>true), "      group size: ", Tuple(length.(x.g)))
end

# function Base.show(io::IO, x::SimpleDataSet)
# 	print(io, "SimpleDataSet of $(length(x.g)) groups:\n", "y: ", x.y, "\ng:", x.g)
# end
