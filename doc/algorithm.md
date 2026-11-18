# Algorithm and experiments

Each pattern has `N` independent standard-normal features and a random label `s ∈ {-1,+1}`. A dataset contains `P` patterns; capacity plots use `alpha = P/N`. The program generates data internally, and `--seed` makes runs repeatable for a fixed toolchain.

Training starts with zero weights and visits patterns in order. For pattern `μ`, the signed potential is `Eμ = sμ(w · xμ)`. If `Eμ <= threshold`, update `wi += sμ xi/N`. The default threshold is zero; with it, a run converges when every pattern has positive potential. Negative thresholds can let nonpositive potentials skip updates, so use zero for standard assignment results. Training stops at convergence or `--max-epochs`. When omitted, the epoch limit is `N * nD`, where `nD` is the configured independent dataset/trial count, not the pattern count `P`. Each swept N gets its own derived limit; an explicit positive override stays fixed.

## Capacity statistics

A capacity study runs `--datasets` independent datasets for each P. Let q be the true training-success probability and `Q_ls = successes/nD`. Then `E[Q_ls]=q` and `Var(Q_ls)=q(1-q)/nD`; for large nD, `Q_ls` is approximately normal with that mean and variance.

The CSV records the plug-in variance estimate `Q_ls_variance_est = Q_ls(1-Q_ls)/nD` and 95% Wilson limits
`(Q_ls+z²/(2nD) ± z√(Q_ls(1-Q_ls)/nD+z²/(4nD²)))/(1+z²/nD)`, with `z=1.96`.

## Theoretical linear separability

For independent Gaussian patterns in general position with random binary labels, Cover's exact probability is
`P_ls = 2^(1-P) Σ[k=0..N-1] binom(P-1,k)`. Equivalently, it is `Pr(X≤N−1)` for `X~Bin(P−1,1/2)`, where `E[X]=(P−1)/2` and `SD(X)=√(P−1)/2`. In particular, `P_ls=1` when `P≤N`.

The CSV also reports the continuity-corrected normal approximation `Φ((2N-P)/√(P-1))` (`P=1` is defined as 1). At fixed load `α=P/N`, the limiting probability is 1 for `α<2` and 0 for `α>2`. The transition width is `O(N^-1/2)`: if `P=2N+b√N+o(√N)`, then `P_ls→Φ(-b/√2)`. Thus `α→2` alone does not determine the limit; the exact sequence `P=2N` has `P_ls=1/2` for every N.

Plots show the empirical Wilson interval, exact and approximate curves, and the `α=2` limit. With the standard zero threshold, finite-budget training success requires separability, so `q≤P_ls`; sampling can nevertheless yield `Q_ls>P_ls`. Negative thresholds weaken that implication.

A strength value `xμ` counts how many updates were caused by pattern μ. The strength histogram includes only patterns from trials that converged.
