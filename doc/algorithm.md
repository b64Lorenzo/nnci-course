# Algorithm and experiment notes

## 1. Data model and notation

Each generated training example consists of an input vector `x^mu` with `N` independent standard-normal components and a label `s^mu` sampled uniformly from `{-1,+1}`. A dataset contains `P` examples. The dimension `N` and pattern count `P` are separate; the load ratio used in capacity plots is:

```text
alpha = P / N
```

A fixed pseudorandom seed makes the generated experiment repeatable for a given build/toolchain. The program generates new datasets for each trial; it does not read an external dataset.

## 2. Rosenblatt perceptron training

The weight vector starts at zero. Examples are visited sequentially in their generated order, cyclically over complete epochs. For an example `(x^mu, s^mu)`, define the signed potential:

```text
E^mu = s^mu * (w dot x^mu)
```

When `E^mu <= threshold`, update every weight component using the implemented rule:

```text
w_i <- w_i + s^mu * x_i^mu / N
```

The default threshold is zero. After an epoch in which updates occurred, the implementation checks all examples and declares convergence only when every signed potential is strictly positive. If an epoch has no updates, it marks the trial converged; with the default threshold of zero, that means every potential is positive. A negative custom threshold can let a non-positive potential avoid an update, so use the default threshold for the standard assignment interpretation. Training ends on convergence or after `nmax` epochs. The final result records the number of epochs, updates, convergence state, final weights, and pattern-indexed strengths. Non-finite potentials or weights are reported as numerical failures.

`--threshold` controls the update condition; the standard assignment run uses the default threshold `0.0`.

## 3. Capacity study

For each P value, the study generates `nD` independent datasets and runs the perceptron on each. A trial is counted as a success only if training converges within `nmax` epochs. The reported empirical success probability is:

```text
Q_ls = successes / nD
```

The output CSV stores the integer `P`, its effective `alpha=P/N`, the trial count, epoch limit, number of successes, success fraction, and seed. The capacity plot shows empirical `Q_ls` and the theoretical general-position curve for random dichotomies:

```text
P_ls(N,P) = 2^(1-P) * sum(C(P-1,k), k=0..N-1)
```

Terms with `k > P-1` contribute nothing. The plot uses `alpha=P/N` on the horizontal axis and success probability on the vertical axis. The finite training limit matters: a dataset that did not converge by `nmax` is recorded as a failure for the empirical estimate even if a longer run might converge. This can make the empirical curve lower than the theoretical curve.

The generated CSV has columns:

```text
N,P,alpha,nD,nmax,successes,Q_ls,seed
```

### N sweep

`--NCapacity START END STEP` runs a capacity study at each requested N and creates one combined plot. Unless P options are supplied, the default P range is recalculated for every N (`p-step=max(1,N/10)`, `p-min=p-step`, `p-max=3N`). Explicit `--p-min`, `--p-max`, and `--p-step` values are interpreted as the same absolute P counts for every N. Thus an N sweep does not reinterpret P as a dimension or silently replace a user-selected pattern range.

## 4. Embedding strength

For each pattern `mu`, `x_mu` starts at zero and increases by one every time that pattern triggers a weight update. It counts updates attributed to that specific pattern; it is not a feature weight, a number of dimensions, or a binary success indicator.

The strength study collects the vector of `x_mu` values only for trials that converged within the epoch limit. The histogram pools the pattern-indexed values from all such successful trials. It therefore describes the distribution of pattern update counts among converged runs, not among failed or unfinished runs. If no trial converges, no histogram is produced and the command reports that there are no strength values to plot.

The strength CSV columns are:

```text
successful_case,mu,x_mu
```

`successful_case` identifies a converged trial, `mu` is the one-based pattern index within that trial, and `x_mu` is its non-negative integer strength.

## 5. Experimental reporting

For a reproducible report, record the executable version and platform along with N, P bounds/step, dataset count, maximum epochs, threshold, and seed. State that the empirical result measures convergence under a finite epoch budget. Do not describe a non-converged run as proof of non-separability. Keep per-N CSVs and the combined N-sweep plot so the plotted curves can be regenerated and inspected.
