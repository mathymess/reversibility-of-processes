# Analyzing reversibility of physical processes with ML

Time irreversibility is a fundamental concept in physics, and the analysis of this property can provide insights into the underlying physical laws that govern the universe.
However, the study of time irreversibility is often limited to mathematical models and computational simulations, and it can be challenging to gain a deeper understanding of the underlying principles.
In this project, we aim to analyze time irreversibility through the lens of neural networks.
The approach would be to compare the performance of the predictive models in both time directions for various physical systems, including Kepler orbital motion, Lorenz attractors and Belousov-Zhabotinsky reaction.
The difference in performance or architecture giving similar performance should indicate the symmetry in the physics laws.

Predicting the trajectory of a dynamical system can be thought of as a time series problem: knowing the position at moments $t_{1}, \ldots, t_{n-1}$, predict the position at time $t_n$.
In this project we use primitive ML to test the following hypothesis: if the process is irreversible, time reversal of the trajectory should affect the difficulty of such prediction.

## Tensorboard history

### 5.1, 20230505, git tag `tensorboard5.1`

Same as `tensorboard5`, but `hidden_layer_size=30` and 10k points with timestep=0.025.
- [double pendulum](https://tensorboard.dev/experiment/L6U5spbYQjOpDdhkAXzjSA/)

In `tensorboard5` the model is probably underfitted, since the loss doesn't improve much, so I tried increasing the model size.

All plots are again similar: starts at 4, reach ~0.9 by epoch 1 and reach ~0.6 at epoch 50.
No difference between `forward` and `backward`.

### 5, 20230428, git tag `tensorboard5`

New system: double pendulum!
It is chaotic, yet perfectly reversible, and our method must indicate that.
Tried 2 different time series samplings, with `hidden_layer_size` in (10,20) and `window_len` in (5,12,25).

1. [double pendulum, 9k points with timestep=0.067, more sparce](https://tensorboard.dev/experiment/cZURcvUaQMap8izA7HaiBw/)
2. [double pendulum, 10k points with timestep=0.025, more dense](https://tensorboard.dev/experiment/LNdqORYLRpCvdee5cnZFNQ/)

All plots are very similar: starts at 4, reach ~1.7 by epoch 5 and reach ~1 at epoch 50.

There's no qualitative difference between two sampling strategies.
The data confirms the hypothesis that the double pendulum is reversible: `forward` and `backward` are identical.

### 4.2, 20230328, git tag `tensorboard4.2`

Rerun `tensorboard4.1` on Kepler with `hidden_layer_size` =10 instead of 5.
- [Kepler](https://tensorboard.dev/experiment/lrZZRKRVRk2VKYRIA5hvqg/)
    - Now that there are way more params, the model probably overfits, getting to 1e-4 loss, but no weird underfitting as with `hidden_layer_size=5`.
    - No clear winner `forward` vs `backward`, as expected

### 4.1, 20230328, git tag `tensorboard4.1`

As in `tensorboard4`, vary `window_len` and `target_len` with `hidden_layer_size` fixed.
Pecularities:
- `chunk_len=window_len+target_len` is kept constant to see how `window_len` affects prediction without changing the total number of trainable parameters in the model (proportional to `chunk_len`).
- Optimizer is still `Adam+ExponentialLR` but changed `gamma` from 0.95 to 0.96 ($0.96^{n_{\text{epoch}}=50} \approx 0.13$ vs $0.95^{n_{\text{epoch}}=50} \approx 0.08$).
- The grid is much sparser (so it's easier to grasp) and now it's hardcoded.

Tensorboards:
- [Kepler](https://tensorboard.dev/experiment/ck9wyrkVQ1mlZ1f2Xon3Hg/)
    - For Lorenz and Belousov-Zhabotinsky `hidden_layer_size=13` and for `Kepler` I used `hidden_layer_size=5`.
    - There are several weird curves that stay at 0.4 after epoch 5, probably need more params.
- [Lorenz](https://tensorboard.dev/experiment/33WgkM1ATly1AKRHkDwi8Q/)
- [Belousov-Zhabotinsky](https://tensorboard.dev/experiment/UkEjV8U8RLaAYadwGnvqDw/)
    - all learning curves are almost identical: abrupt loss drop after the first epoch, then `forward` slowly and steadily wins, which you can better see in the log scale

### 2.1, 20230326, git branch `tensorboard2.1`

Rerun `tensorboard2` on Belousov-Zhabotinsky after I changed the dataset so that it only includes the first period of the periodic motion.
It used to include about 20 identical periods, and I thought it was wrong.

- [Belousov-Zhabotinsky](https://tensorboard.dev/experiment/E4jbjQP4Tdak7MbvEXWyyg/)

Observations:
- For some reason, learning curves are much smoother than for `tensorboard2`. It would be pointless to add a scheduler `ExponentialLR`.
- For `hidden_layer_size` equal to 1 or 5, weird things happen, so I assume model needs more parameters to learn.
- For `hidden_layer_size` equal to 9,13,17 `forward` quickly reaches 1e-3 loss, while `backward`'s loss increases and then falls back (why?).
- `backward` has reliably greater loss than `forward` -- the process is "irreversible"

It is unobvious whether or not shrinking the dataset to 1 period was a good idea.

### 4, 20230320, git tag `tensorboard4`

Vary `window_len` and `target_len` at `hidden_layer_size=13` with (`torch.optim.Adam` + `torch.optim.lr_scheduler.ExponentialLR(gamma=0.95)`).

- [Lorenz](https://tensorboard.dev/experiment/9mmpTyOXQ1Gf4k03dadWZg/)

Observations:
- Too many pictures, hard to make conclusions + computation takes too long.
- `backward` has greater loss that `forward`, but often insignificantly. Need a closer look with fewer pictures.
- It might be better to vary `window_len` and `target_len` and keep their sum (proportional to the total amount of parameters in the model) constant.
- The bigger `target_len` is, the greater the typical loss values are.
I average the loss over the train dataset, but not over each target point.
- For `window_len>36` and `target_len > 0.6*window_len`, loss goes, very roughly speaking, from 90 down to 30.
Probably underfitting, probably due to `ExponentialLR` dying out too fast.
- Consider `window_len=76`, `target_len=31`.
This amounts to `chunk_len=107`, which is 1% of the 10000 points in the original time series.
If you look at the plot in `dataset_review.ipynb`, this is a huge `chunk_len`.
If you look in `model_review.ipynb`, with `size=13` the total number of trainable parameters in the model is about 4.5k, half the training dataset size.
This is to say, I should've stopped at `window_len=30`.

### 3.1, 20230320, git tag `tensorboard3.1`

Same as `tensorboard3`, except
- added 4th optimizer into comparison: `torch.optim.RMSprop` + `torch.optim.lr_scheduler.ExponentialLR(gamma=0.95)` 
- changed `hidden_layer_size` from 10 to 16.

Other parameters remain `window_len=30`, `target_len=1`.

- [Lorenz](https://tensorboard.dev/experiment/135AOEnBQDeraFPTwzFXQw/)
    - `Adam` is a bit noisy
    - `Adam+ExponentialLR` is very smooth, increase `gamma=0.95` to make it less smooth ($0.95^{n_{\text{epoch}}=50} \approx 0.07$)
    - `RMSprop` -- model doesn't learn, too noisy
    - `RMSprop+ExponentialLR` roughly same as Adam, a bit noisy

### 3, 20230319, git tag `tensorboard3`

I compare three different optimizers, all with default parameters:
1. `torch.optim.Adam` (was used in all runs before)
2. `torch.optim.RMSprop` (turned out to be too noisy)
3. `torch.optim.Adam` + `torch.optim.lr_scheduler.ExponentialLR(gamma=0.95)` (maybe optimal, maybe too smooth)

Other parameters are fixed: `window_len=30`, `hidden_layer1_size=hidden_layer2_size=10`, `target_len=1`.

- [Lorenz](https://tensorboard.dev/experiment/V00WLnJQTMKrnZR76JkRKg/)

### 2, 20230313, git tag `tensorboard2`

Test dataset is the same as train to avoid randomization and sampling bias observed in `tensorboard1.1`.
For each system, there are ~50 learning curves with hidden_layer_size going from 1 to 20.
This corresponds to the total number of parameters in `ThreeFullyConnectedLayers` ranging from ~0.1k to ~2k.
I rerun the same learning process 3 times, each labeled by one of the letters `a,b,c` to make up for some randomness due to randomized batching in `torch.utils.data.DataLoader` and initial weights.
An observation: for small `hidden_layer_size`, loss usually stops at value > 10, implying the model doesn't learn.

- [Kepler](https://tensorboard.dev/experiment/lQ62rBh6TDG9cDSg0s8lDQ/)
    - size 1-4: weird stuff, too few params
    - size 10-20: best fit after 5-10 epochs, crazy noise with 1e-2 loss afterwards
    - size 5-9: a bit noisy, something in between.
    - no clear winner `forward` vs `backward`
- [Lorenz](https://tensorboard.dev/experiment/NNyGP2F0T3KHZbvurDLvsw/)
    - size 1-4: weird stuff, too few params
    - size 5-8: very smooth
    - size 5-20: `backward` has greater loss about 80% of the time.
- [Belousov-Zhabotinsky](https://tensorboard.dev/experiment/UmfOElNZRRqdd3kt9LbzKg/)
    - size 1-3: weird stuff, too few params
    - size 6-20: noisy, but `backward` is strictly greater than `forward`, and also much noisier

### 1.1, 20230312, git tag `tensorboard1.1`

Redo the exact same plots with few minor fixes.

- [Kepler](https://tensorboard.dev/experiment/NmioEasRR023gljiQKdsyQ/)
- [Lorenz](https://tensorboard.dev/experiment/jbwsyZyPT6iBbJPfB3QEdw/)
- [Belousov-Zhabotinsky](https://tensorboard.dev/experiment/LyNtPio7TdSri93mRq1l3g/)

### 1, 20230306, git tag `tensorboard1`

For each of three physical systems, I vary (1) the hidden layer size at fixed `window_len` and (2) `window_len` and `shift_ratio` at fixed hidden layer size.
`shift_ratio` defines which part of the periodic trajectory we consider to be the test and which to be the train data.

- [Kepler](https://tensorboard.dev/experiment/MOjL9KUlR0ik1Dvr4au7CQ/)
- [Lorenz](https://tensorboard.dev/experiment/jAuEyWfpQCWJgxiwlZnBqg/)
- [Belousov-Zhabotinsky](https://tensorboard.dev/experiment/T8aXeU7DSRClvkvARzdjkg/)

The somewhat chaotic results for (2) show that `shift_ratio` is important.
If you reveal only a region of the periodic orbit for training, the remaining [test] region might be qualitatively different from the training one, and it's unreasonable to expect that the model will make accurate predictions about the hidden part of the orbit.
