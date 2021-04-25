# TODO

- For the Dirichlet process prior (DPP), use \alpha = 1.817 like Gopalan & Berry.
- Look at overlaying multiple priors into one figure.
- Remove connecting axes in figure of priors?

- Sample a model with DPP.
- Make example where DPP is inconsistent, i.e., DPP does not retrieve the 'true' partition.
- Ideally Beta-binomial is consistent.
- How quickly do we recover the true model?
- Make a plot of Bayes factor using BF_M from JASP to demonstrate convergence rate.

- Use a Beta-Binomial distribution for determining a new value and the categorical weights from the Dirichlet Process. That should reintroduce the "rich get richer" property.

# Plots
- 3 by 2 of the Scott & Berger plot for all 3 priors.
- Expected number of clusters.


# 24-03-2021
- numerical precision issue S & B figure 1, 2
- start on simulations

# Sprint plan (22-04 & 23-04)

# Writing
- Clear up notation

# Plots
- comparison of priors
- convergence rate plots




# Scott & Berger plot

only BetaBinomial & DPP

add a couple more parameter values for BB & DPP

BB(1, 1)
BB(2, 2)
BB(1, 2)

DPP(1)
DPP(1.887)
DPP(5)

# convergence simulation

- assess if we converges to true value given a particular prior
- behavior as the model space grows
- study consistency results for linear regression

for us, k = 5 and grow n, plot posterior probability of true model

# performance
- make precomputed tables for the r-stirling numbers and stirlings1



# planning for the next weeks

