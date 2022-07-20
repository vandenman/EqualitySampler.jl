using Documenter, EqualitySampler

makedocs(
	sitename="My Documentation",
	modules  = [EqualitySampler],
    format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
)

deploydocs(;
    repo = "github.com/vandenman/EqualitySampler.git",
    versions = ["stable" => "v^", "v#.#", "dev" => "main"]
)