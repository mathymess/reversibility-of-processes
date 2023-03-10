# Analyzing reversibility of physical process with ML

Time irreversibility is a fundamental concept in physics, and the analysis of this property can provide insights into the underlying physical laws that govern the universe.
However, the study of time irreversibility is often limited to mathematical models and computational simulations, and it can be challenging to gain a deeper understanding of the underlying principles.
In this project, we aim to analyze time irreversibility through the lens of neural networks.
The approach would be to compare the performance of the predictive models in both time directions for various physical systems, including Kepler orbital motion, Lorenz attractors and Belousov-Zhanotinsky reaction.
The difference in performance or architecture giving similar performance should indicate the symmetry in the physics laws.

Predicting the trajectory of a dynamical system can be thought of as a time series problem: knowing the position at moments $t_{1}, \ldots, t_{n-1}$, predict the position at time $t_n$.
In this project we use primitive ML to test the following hypothesis: if the process is irreversible, time reversal of the trajectory should affect the difficulty of such prediction.

## Tensorboard history

### commit `1f4f3fa`

- Kepler `traintest_kepler.py`: https://tensorboard.dev/experiment/MOjL9KUlR0ik1Dvr4au7CQ/
- Belousov-Zhabotinsky `traintest_belousov_zhabotinsky.py`: https://tensorboard.dev/experiment/T8aXeU7DSRClvkvARzdjkg/
- Lorenz `traintest_lorenz.py`: https://tensorboard.dev/experiment/jAuEyWfpQCWJgxiwlZnBqg/

For each of three physical systems, I vary (1) the hidden layer size at fixed `window_len` and (2) `window_len` and `shift_ratio` at fixed hidden layer size.
`shift_ratio` defines which part of the periodic trajectory we consider to be the test and which to be the train data.
The somewhat chaotic results for (2) show that `shift_ratio` is important.
If you reveal only a region of the periodic orbit for training, the remaining [test] region might be qualitatively different from the training one, and it's unreasonable to expect that the model will make accurate predictions about the hidden part of the orbit.
