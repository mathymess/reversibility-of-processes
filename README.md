# Analyzing reversibility of physical process with ML

Time irreversibility is a fundamental concept in physics, and the analysis of this property can provide insights into the underlying physical laws that govern the universe.
However, the study of time irreversibility is often limited to mathematical models and computational simulations, and it can be challenging to gain a deeper understanding of the underlying principles.
In this project, we aim to analyze time irreversibility through the lens of neural networks.
The approach would be to compare the performance of the predictive models in both time directions for various physical systems, including Kepler orbital motion, Lorenz attractors and Belousov-Zhanotinsky reaction.
The difference in performance or architecture giving similar performance should indicate the symmetry in the physics laws.

Predicting the trajectory of a dynamical system can be thought of as a time series problem: knowing the position at moments $t_{1}, \ldots, t_{n-1}$, predict the position at time $t_n$.
In this project we use primitive ML to test the following hypothesis: if the process is irreversible, time reversal of the trajectory should affect the difficulty of such prediction.

## Tensorboard history

### 3, 20230313, git tag `tensorboard3`

Test dataset is the same as train to avoid randomization and sampling bias observed in `tensorboard2`.
For each system, there are ~50 learning curves with hidden_layer_size going from 1 to 20.
This corresponds to the total number of parameters in `ThreeFullyConnectedLayers` ranging from ~0.1k to ~2k.
I rerun the same learning process 3 times, each labeled by one of the letters `a,b,c` to make up for some randomness due to randomized batching in `torch.utils.data.DataLoader` and initial weights.
An observation: for small hidden_layer_size, loss usually stops at value > 10, implying the model doesn't learn.

- [Kepler](https://tensorboard.dev/experiment/lQ62rBh6TDG9cDSg0s8lDQ/)
- [Belousov-Zhabotinsky](https://tensorboard.dev/experiment/GNzNl6v8RgO9HkwWk8S5QA/)
- [Lorenz](https://tensorboard.dev/experiment/NNyGP2F0T3KHZbvurDLvsw/)

### 2, 20230312, git tag `tensorboard2`

Redo the exact same plots with few minor fixes.

- [Kepler](https://tensorboard.dev/experiment/NmioEasRR023gljiQKdsyQ/)
- [Belousov-Zhabotinsky](https://tensorboard.dev/experiment/LyNtPio7TdSri93mRq1l3g/)
- [Lorenz](https://tensorboard.dev/experiment/jbwsyZyPT6iBbJPfB3QEdw/)

### 1, 20230306, git tag `tensorboard1`

- [Kepler](https://tensorboard.dev/experiment/MOjL9KUlR0ik1Dvr4au7CQ/)
- [Belousov-Zhabotinsky](https://tensorboard.dev/experiment/T8aXeU7DSRClvkvARzdjkg/)
- [Lorenz](https://tensorboard.dev/experiment/jAuEyWfpQCWJgxiwlZnBqg/)

For each of three physical systems, I vary (1) the hidden layer size at fixed `window_len` and (2) `window_len` and `shift_ratio` at fixed hidden layer size.
`shift_ratio` defines which part of the periodic trajectory we consider to be the test and which to be the train data.
The somewhat chaotic results for (2) show that `shift_ratio` is important.
If you reveal only a region of the periodic orbit for training, the remaining [test] region might be qualitatively different from the training one, and it's unreasonable to expect that the model will make accurate predictions about the hidden part of the orbit.
