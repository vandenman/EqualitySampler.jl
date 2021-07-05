## Chores
- [x] Upload ISBA slides to GitHub
- [x] Email Han

## Julia

### simulations/visualizePriors.jl
- [x] Legend in  in the topright of the top row

### simulations/multipleComparisonPlot.jl
- [ ] axes lines should not exceed the limits (y: (0, 1), x: (2, 10))
- [x] odd numbers on the x-axis
- [x] try no space between prior family and parameters
- [x] hashtag symbol for no. groups in legend
- [x] legend slighlty left
- [x] x-axis title should be Number of groups

### simulations/meansModel_simulation_convergence.jl
- [x] More repetitions, at least 50! After reparametrizing the model maybe?

### simulations/meansModel_analyze_convergence_results.jl
- [x] One big figure for the appendix
- [x] Labelling went wrong! BetaBinomial should be DPP in right columns

### demos/proportions.jl
- [ ] same issue where axes lines should not exceed the limits (y: (0, 1), x: (2, 10))



### General speed improvements:

- [ ] intterupt stopped at:
logpdf_model_distinct at /home/dvdb/github/EqualitySampler/src/multivariateUrnDistributions.jl:192
logpdf_model at /home/dvdb/github/EqualitySampler/src/multivariateUrnDistributions.jl:48 [inlined]
logpdf at /home/dvdb/github/EqualitySampler/src/multivariateUrnDistributions.jl:52 [inlined]
logposterior at /home/dvdb/github/EqualitySampler/simulations/meansModel_Functions.jl:294

examine whether DPP prior is not horribly slow!

- [ ] Figure out why DPP and BB (mainly DPP) are so much slower than Uniform!

- [ ] play with JETTest.jl

- [ ] benchmark logpdf of models somehow?

- [ ] detect whether a run contains many errors and store this in the results?

- [ ] consider looping over the priors within each run
  - [ ] possibly loading the old file to see if every prior is already there
