using Turing

@model function dummymodel(dim::Int)
	
	dummies = Matrix{Int}(undef, dim - 1, 2)
	for j in 1:2, i in 1:dim - 1
		dummies[i, j] ~ Categorical(ones(Float64, i + 1) / (i + 1))
	end
end

model_dummy = dummymodel(3)
samples_dummy = sample(model_dummy, PG(20, :dummies), 10_000)

values = Int.(samples_dummy[samples_dummy.name_map.parameters].value.data)

transformed_vals = values[:, [1, 3]]
for i in axes(transformed_vals, 1)
	transformed_vals[i, :] .= (transformed_vals[i, :] .% values[i, [2, 4]]) .+ 1
end

