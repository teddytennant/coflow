# coflow

Contextual flow-matching adaptive step selection from https://arxiv.org/abs/2610.03202: NeuralUCB picks an Euler step budget from prompt features and a trajectory-variation reward.

## Install

```
pip install -e .
```

Tests need the optional extra: `pip install -e ".[test]"`.

## Run

```
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python demo.py
```

The demo trains the bandit on a 2D synthetic velocity field and prints mean greedy steps for easy prompts and hard prompts.

## Method

Forward Euler on [0, 1] with budget alpha uses t_k = k / alpha and x_{k+1} = x_k + (1 / alpha) * v(x_k, t_k). Trajectory variation is

vhat(x, alpha) = (1 / (alpha - 1)) * sum_{k=1}^{alpha-1} ||v(x_{t_{k+1}}, t_{k+1}) - v(x_{t_k}, t_k)||_1 / max(||v(x_{t_k}, t_k)||_1, 1e-8)

The reward is r(x, alpha) = max(0, (1 - vhat(x, alpha)) - mu * alpha), with mu = 1 / max(A). Because mu * max(A) = 1, the largest arm has reward 0. The context is the 8 handcrafted features below, concatenated with a 32-dimensional hash embedding. The reward model is a two-layer MLP (one hidden layer of width 64, ReLU, linear scalar head). Its input is concat(phi, alpha / max(A)).

Handcrafted features, in table order: word count min(#words / 50, 1), type-token ratio |unique lowercased tokens| / #tokens, mean word length min(mean(|w|) / 10, 1), numeric fraction, object density, attribute density, punctuation density min(#punctuation / (2 * #words), 1), syllable complexity min(mean syllables / 4, 1). Words are alphanumeric tokens, with an internal apostrophe kept inside the token. Object density treats non-stopword tokens that contain a letter as nouns. Attribute density treats tokens with common adjective suffixes (able, ible, al, ful, ous, ive, less, ish, ic, ary, and similar) as adjectives. A token can count toward both. There is no POS tagger.

The hash embedding lowercases the prompt, counts character 3-grams into 512 buckets with FNV-1a, and multiplies by a Gaussian matrix drawn from numpy.random.default_rng(0). The result is L2-normalized. An empty prompt is the zero vector. Other strings shorter than 3 characters are padded with spaces.

NeuralUCB starts from V_0 = lambda I with lambda = 1. It pulls each arm once, then runs a short epsilon-greedy phase (epsilon 0.1, one pass over the arms), then selects alpha = argmax_a [f(phi, a) + beta_t * sqrt(g^T V^{-1} g)] with g = grad_theta f at (phi, a). beta_t = beta0 * sqrt(log(t + 1)) with beta0 = 0.5 and natural log. The paper cites Zhou et al. and does not give a numeric beta. Every P steps (default P = 10) the replay buffer is fit by MSE with Adam at learning rate 5e-3. The paper uses 20 epochs; `epochs` is an argument and tests may use fewer. Default arms are {10, 15, 20, 25, 30, 35, 40, 45, 50}.

The paper's Euler theorem is an O(1/K) global error bound when the velocity is Lipschitz and the second derivative of the flow is bounded. On dx/dt = -x the exact endpoint is x0 * exp(-1). Forward Euler error decreases with K, and K * error(K) stays within a small constant factor for K in {5, 10, 20, 40}.

The synthetic field is v(x, t; difficulty) = (target - x) + difficulty * sin(2 * pi * freq * t) * u, with target (1, 0), u = (0, 1), and freq = 0.75. Difficulty is a logistic gate on max(word count, numeric fraction), scaled by 12, so a high word count or a high numeric fraction makes the field rough. A smooth field has low variation at a small budget, so the paper reward prefers a small budget. A rough field does not: variation stays large at 10 steps and drops enough by an intermediate budget that (1 - vhat) beats mu * alpha. The demo uses 64 rounds and refits every 8 steps so a seeded run separates easy and hard prompts. Reported means are greedy argmax of the learned reward. The UCB bonus is still on during data collection, and on easy prompts it can be as large as the reward gap.

## What does not match the paper

The semantic embedding is the seed-0 hash projection above, not all-MiniLM-L6-v2. Nouns and adjectives are the suffix and stopword approximations above, not spaCy or NLTK tags.

beta_t = 0.5 * sqrt(log(t + 1)) is not in the paper. The paper cites Zhou et al. and does not give a numeric beta. Epsilon, the length of the epsilon-greedy phase, and the update interval P are not in the paper. Defaults are 0.1, one pass over the arms after the cold start, and P = 10. The replay buffer keeps the full history. The paper text says the buffer size is P, while the algorithm minimizes over all observed data.

The velocity field is the 2D synthetic field above, not FLUX or BAGEL. Difficulty gain 12, gate threshold 0.25, and frequency 0.75 are chosen so the synthetic reward prefers few steps on easy prompts and many steps on hard prompts. They are not paper hyperparameters. Integer frequencies can alias to zero on a coarse Euler grid. Frequency 0.75 is not an integer. It is low enough that a rough field has a higher paper reward at an intermediate budget than at 10 steps. A higher frequency keeps relative variation above 1 - mu * alpha at every arm, so the clipped reward is flat.
