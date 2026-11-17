# Algorithm and experiments

Each pattern has `N` independent standard-normal features and a random label `s ∈ {-1,+1}`. A dataset contains `P` patterns; capacity plots use `alpha = P/N`. The program generates data internally, and `--seed` makes runs repeatable for a fixed toolchain.

Training starts with zero weights and visits patterns in order. For pattern `μ`, the signed potential is `Eμ = sμ(w · xμ)`. If `Eμ <= threshold`, update `wi += sμ xi/N`. The default threshold is zero; with it, a run converges when every pattern has positive potential. Negative thresholds can let nonpositive potentials skip updates, so use zero for standard assignment results. Training stops at convergence or `--max-epochs`.

A capacity study runs `--datasets` independent datasets for each P. The measured success rate is `Qls = successes/datasets`; a run that misses the epoch limit counts as a failure. The plot compares this estimate with the theoretical random-dichotomy curve. Treat the measured curve as an estimate, not proof that failed datasets are inseparable.

A strength value `xμ` counts how many updates were caused by pattern μ. The strength histogram includes only patterns from trials that converged.
