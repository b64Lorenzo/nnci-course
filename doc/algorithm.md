# Algorithm and experiments

Each pattern has `N` independent standard-normal features and a random label `s ∈ {-1,+1}`. A dataset contains `P` patterns; capacity plots use `alpha = P/N`. The program generates data internally, and `--seed` makes runs repeatable for a fixed toolchain.

## Training rules

Both backends start with zero weights, use finite threshold `c` (default zero), and signed potentials `Eμ = sμ(w · inputμ)`. A solution in the final permitted epoch counts as success. The default epoch limit is `N * nD`, where `N` is feature dimensions and `nD` is datasets/trials, not pattern count `P`. Each swept N gets its own limit; an explicit positive `--max-epochs` stays fixed.

### CPU: online Rosenblatt

CPU training, including `--parallelize`, visits patterns in order. If the current `Eμ <= c`, it immediately updates `w += sμ inputμ/N`; later patterns see the changed weights. An epoch with no updates converges immediately, even for a negative threshold. Otherwise all final potentials must be strictly positive. This existing CPU behavior is unchanged; use `c=0` for standard assignment results.

### CUDA: synchronous batch Hebbian learning

Since v0.3.2, `--cuda` uses this recurrence, not the online CPU algorithm:

1. Freeze all `Eμ(t)` from `w(t)` and select `A(t) = {μ : Eμ(t) <= c}`.
2. Sum the selected Hebbian vectors in parallel and apply **one** weight update:
   `w(t+1) = w(t) + (1/N) * sum_{μ in A(t)} sμ inputμ`.
3. Recompute every `Eμ(t+1)` in parallel. Stop iff `minμ Eμ(t+1) > 0`; otherwise repeat until the epoch limit.

The learning rate is `1/N`: divide the total sum by N, not by `1/N`, P, or the selected-pattern count. Each selected pattern contributes once to its strength and to the update counter per epoch. An empty selection or cancelling sum does not imply convergence: if any potential is zero or negative, training continues to the limit. Non-finite sums, weights or potentials are errors.

Batch and online learning can produce different weights, strengths, epochs and capacity estimates. CUDA reductions use FP64 with a fixed reduction tree; their floating-point addition order differs from a scalar reference. Seeded reruns remain deterministic for a fixed toolchain/device.

## Capacity statistics

A capacity study runs `--datasets` independent datasets for each P. Let q be the true training-success probability and `Q_ls = successes/nD`. Then `E[Q_ls]=q` and `Var(Q_ls)=q(1-q)/nD`; for large nD, `Q_ls` is approximately normal with that mean and variance.

The CSV records the plug-in variance estimate `Q_ls_variance_est = Q_ls(1-Q_ls)/nD` and 95% Wilson limits
`(Q_ls+z²/(2nD) ± z√(Q_ls(1-Q_ls)/nD+z²/(4nD²)))/(1+z²/nD)`, with `z=1.96`.

## Theoretical linear separability

For independent Gaussian patterns in general position with random binary labels, Cover's exact probability is
`P_ls = 2^(1-P) Σ[k=0..N-1] binom(P-1,k)`. Equivalently, it is `Pr(X≤N−1)` for `X~Bin(P−1,1/2)`, where `E[X]=(P−1)/2` and `SD(X)=√(P−1)/2`. In particular, `P_ls=1` when `P≤N`.

The CSV also reports the continuity-corrected normal approximation `Φ((2N-P)/√(P-1))` (`P=1` is defined as 1). At fixed load `α=P/N`, the limiting probability is 1 for `α<2` and 0 for `α>2`. The transition width is `O(N^-1/2)`: if `P=2N+b√N+o(√N)`, then `P_ls→Φ(-b/√2)`. Thus `α→2` alone does not determine the limit; the exact sequence `P=2N` has `P_ls=1/2` for every N.

Plots show the empirical Wilson interval, exact and approximate curves, and the `α=2` limit. With the standard zero threshold, finite-budget training success requires separability, so `q≤P_ls`; sampling can nevertheless yield `Q_ls>P_ls`. Negative thresholds weaken that implication for the online CPU rule only; CUDA always requires strict positive final potentials.

A strength value `xμ` counts how many updates were caused by pattern μ. The strength histogram includes only patterns from trials that converged.
