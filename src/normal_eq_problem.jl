using Turing, DynamicPPL, FillArrays # Turing v0.14.10

@model function normal_mean_variance(x, useDiscreteUniform, useDirichlet)

    _, k = size(x)
    m ~ filldist(Normal(0, 1), k)

    if useDirichlet
        tau ~ InverseGamma(1, 1)
        r ~ Dirichlet(ones(k))
        s = 1 ./ sqrt.(k * tau .* r)
    else
        s ~ filldist(InverseGamma(1, 1), k)
    end

    equal_indices = ones(Int, k)
    if useDiscreteUniform
        for i in 1:k
            # a bad prior, but let's forget about that for now
            equal_indices[i] ~ DiscreteUniform(1, i)
        end
    else
        # a better prior
        equal_indices[1] ~ Categorical([1])
        equal_indices[2] ~ Categorical([2/5, 3/5])
        equal_indices[3] ~ Categorical(equal_indices[2] == 1 ? [.5, 0, .5] : 1/3 .* ones(3))
    end

    s_eq = s[equal_indices]
    for i in axes(x, 1)
        x[i, :] ~ MvNormal(m, s_eq)
    end
    return (m, s_eq)
end

function fitmodel(x::AbstractMatrix, useDiscreteUniform::Bool, useDirichlet::Bool)

    k = size(x)[2]
    spl = Gibbs(
        useDirichlet ? HMC(0.05, 10, :m, :t, :r) : HMC(0.05, 10, :m, :s),
        PG(20, :equal_indices)
    )
    m = normal_mean_variance(x, useDiscreteUniform, useDirichlet)

    chn = sample(m, spl, 3_000, discard_initial = 2_000);
    gen = generated_quantities(m, chn);

    idx_eq = filter(x->startswith(string(x), "equal_indices["), chn.name_map.parameters);
    eqs = reshape(Int.(chn[idx_eq].value.data), (size(chn)[1], k));

    models = mapslices(x->join(string.(x)), eqs, dims = 2);
    count_models = Dict{String, Int}();
    for i in axes(models, 1)
        haskey(count_models, models[i]) ? count_models[models[i]] += 1 : count_models[models[i]] = 1
    end

    probs_equalities = zeros(k, k);
    for row in eachrow(eqs)
        for j in eachindex(row)
            idx = j .+ findall(==(row[j]), row[j+1:end])
            probs_equalities[idx, j] .+= 1.0
        end
    end
    probs_equalities .= probs_equalities ./ size(eqs)[1];

    return count_models, probs_equalities, m, chn

end

# generate data
D = MvNormal([1.0, 3.0, 5.0], [1.0, 3.0, 5.0])
x = permutedims(rand(D, 100));

count00, probs00, m00, chn00 = fitmodel(x, false, false)
count10, probs10, m10, chn10 = fitmodel(x, true, false)
count01, probs01, m01, chn01 = fitmodel(x, false, true)
count11, probs11, m11, chn11 = fitmodel(x, true, true)


k = size(x)[2]
m = normal_mean_variance(x, false, true) # it's the Dirichlet!

spl = Gibbs(HMC(0.05, 10, :t, :r, :s), PG(20, :equal_indices))
chn = sample(m, spl, 3_000, discard_initial = 2_000);

gen = generated_quantities(m, chn);



idx_eq = filter(x->startswith(string(x), "equal_indices["), chn.name_map.parameters);
eqs = reshape(Int.(chn[idx_eq].value.data), (size(chn)[1], k));

models = mapslices(x->join(string.(x)), eqs, dims = 2);
count_models = Dict{String, Int}();
for i in axes(models, 1)
    haskey(count_models, models[i]) ? count_models[models[i]] += 1 : count_models[models[i]] = 1
end
count_models

probs_equalities = zeros(k, k);
for row in eachrow(eqs)
    for j in eachindex(row)
        idx = j .+ findall(==(row[j]), row[j+1:end])
        probs_equalities[idx, j] .+= 1.0
    end
end
probs_equalities .= probs_equalities ./ size(eqs)[1];
probs_equalities