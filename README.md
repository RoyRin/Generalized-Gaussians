# Generalized Gaussians

## Abstract:
Differential privacy (DP) is obtained by randomizing a data analysis algorithm, which necessarily introduces a tradeoff between its utility and privacy. Many DP mechanisms are built upon one of two underlying tools: Laplace and Gaussian additive noise mechanisms. We expand the search space of algorithms by investigating the Generalized Gaussian (GG) mechanism, which samples the additive noise term $x$ with probability proportional to `e^(-| x |/σ)^β` for some `β >= 1`. The Laplace and Gaussian mechanisms are special cases of GG for `β=1` and `β=2` respectively. 

In this work, we prove that all members of the GG family satisfy differential privacy, and provide an extension of an existing numerical accountant (the PRV accountant) for these mechanisms. We show that privacy accounting for the GG Mechanism and its variants is dimension independent, which substantially improves computational costs of privacy accounting. 

We apply the GG mechanism to two canonical tools for private machine learning, PATE and DP-SGD; we show empirically that `β` has a weak relationship with test-accuracy, and that generally `β=2` (Gaussian) is nearly optimal. This provides justification for the widespread adoption of the Gaussian mechanism in DP learning, and can be interpreted as a negative result, that optimizing over `β` does not lead to meaningful improvements in performance.



# Code Note: 
`Opacus-PRV` is the `Opacus` (https://github.com/pytorch/opacus) library forked in mid-2021; with modifications made.

```
data_aware_dp/
├── __init__.py
├── datasets.py
├── histogram_computation.py
├── histogram_evaluation.py - Code for evaluating histograms with generalized guassians
├── ml_training.py
├── ml_utils.py
├── models.py
├── pate_dadp.py
├── plotting.py
├── rdp.py
├── sampling.py
├── train_ml_cli.py
├── utils.py
└── wide_resnet.py
```

# terminology

There are several terms for the generalized gaussian distribution; in the code it is sometimes referred to as `beta exponenital`, `generalized gaussians`, or `exponential power distribution`.
