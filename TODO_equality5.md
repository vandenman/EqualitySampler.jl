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


### Figure 2
- [x] Reorder DPP, Uniform, BetaBinomial
- [x] No box around the legend
- [x] Left panels need y-axis labels
- [x] Bigger font size
- [x] Switch color for DPP so orange line is decreasing
- [x] Add "Prior" to the column titles
- [x] Column title "BetaBinomial" -> "Beta-Binomial"
- [x] Drop partition numbers (x-axis tick labels)

### Figure 3
- [x] "BetaBinomial" -> "Beta-Binomial"
- [x] "Gopalan Berry" -> "Gopalan & Berry" (ampersand)

### Figure 4 + Figure 5
- [?] check zero vs non-zero? What was this about?
- [x] adjust y-axis limits to (0, .5)
- [x] use mean instead of median
- [ ] For manuscript figure, put No. parameters left and don't repeat prior in the title (basically mimic facet_grid)
- [ ] create nicer figures for true inequalities (power) and all other options

### Scott & Berger plot
- [x] \alpha = 0.5
- [x] make figure and add it to the overleaf

### Add proportion demo
- [ ] Review all the comments in the manuscript

### Appendix
- [x] "BetaBinomial" -> "Beta-Binomial"


## 11-08-2021
- [ ] For all plots
	- [ ] Change order to DPP, beta-binomial, uniform
	- [ ] Increase line width in all plots
	- [ ] Ensure digits have 2 decimals, so 0.50 and not 0.5, 1.00 and not 1.0
- [ ] Figure 2 (visualizePriors)
	- [x] change order
	- [ ] Legend: increase font size and decrease the symbol size
	- [x] Legend: "Gopalan Berry" -> "Gopalan & Berry"
	- [x] double check colors in bottom plot vs top plot and if they match, mainly for DPP
- [ ] Figure 3 (Scott & Berger)
	- [ ] Look at most ??? what did I write here?
	- [x] change order
	- [x] y-axis label should be odds! Copy text from Scott & Berger
	- [x] do Beta-binomial (\alpha = k, \beta = 1)
	- [x] legend should be "1 inequality added", "2 inequalities added", "5 ...", "10 ..."
	- [x] put legend in Beta-binomial plot (after reordering)
	- [x] double check if DPP is not flipped!
	- [x] leave top row, let it go to K = 20.
	- [x] rescale y-axis for Uniform in both bottom and top row.
	- [x] do A-B thingy like in Scott & Berger
	- [x] see if the DPP alpha value determines the asymptote.
- [ ] Figure 5
	- [x] Legend title
	- [x] Legend labels
	- [x] Order of columns is incorrect
	- [x] rename parameters to $K$ = 5
	- [x] x-axis titles and y-axis titles are missing
	- [x] rotate x-axis tick labels by 30 degrees
	- [x] Bonus: make y-axis go to 0.2 where possible
