# Random-init floor stability (excerpt)

Verbatim from road-to-robotics `projects/jepa-distill-edge/README.md` lines 82-84 @ fb68283.

Read before relying on it: the random-init W1 term is not stable across machines. It was 1.4383 on the first §30
instance (`results/gap5.json`), 0.7974 here and 0.8042 on the Orin; W1-scratch and the teacher agree to about 0.001.
The ridge fit warns of an ill-conditioned matrix (`gap5_pooled_logs/gap5.log`), so the floor depends on the BLAS build.
