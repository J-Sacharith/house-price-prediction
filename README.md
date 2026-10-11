# House Price Prediction with Neural Networks

Predicting the sale price of residential homes in Ames, Iowa (1,460 homes, 79 features) with
a TensorFlow/Keras multilayer perceptron, built as a disciplined, reproducible workflow
following Andrej Karpathy's *[A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)*:

> **Data → skeleton + dumb baselines → overfit → regularize → tune**

The goal was not the lowest possible error but a pipeline that can be trusted, and an honest
answer to whether deep learning was the right tool.

## Key finding

**A plain linear regression matched — and slightly beat — the tuned neural network.**

| Model (validation set) | MAE | RMSE | RMSE (log) |
|---|---:|---:|---:|
| Mean predictor | $59,931 | $88,271 | 0.433 |
| **Linear regression** | **$15,075** | **$22,907** | **0.132** |
| Random forest | $17,374 | $29,666 | 0.146 |
| NN-A plain (overfits) | $18,035 | $26,732 | 0.152 |
| NN-B regularized *(final NN)* | $16,808 | $29,788 | 0.142 |
| NN-C tuned (Adam) | $17,575 | $30,803 | 0.147 |

The final network's MAE is about **11 % of the median home price** ($154,150). On ~1.5k rows of
tabular data, deep learning added complexity without a payoff, which is the textbook case for
preferring classical ML. The decisions that mattered most were the `log1p` target transform,
leakage-free preprocessing and regularization with early stopping; architecture tweaks
mattered far less.

## What the notebook covers

1. **Data understanding**: feature types, missing values (NA = "feature absent" for 15
   columns), target skew (1.88 → 0.12 after `log1p`), correlations, outliers
2. **Baselines**: mean/median predictors, linear regression, random forest
3. **Sanity checks**: loss at initialization, overfitting a tiny batch, no-leakage
   confirmation, inspecting exactly what enters the network
4. **Network development**: overfit → regularize (dropout, L2, batch norm, early stopping)
   → tune (optimizer, depth)
5. **Evaluation in business terms**: error as % of median price, error by price tier
6. **Explainability**: permutation importance, residual diagnostics, worst predictions
7. **Deployment and monitoring**: saved model + preprocessor + metadata, reload-and-predict
   demo, covariate/label/concept drift plan

## Run it

```bash
pip install -r requirements.txt
# download train.csv and test.csv from Kaggle into data/ (see data/README.md)
jupyter notebook house_price_prediction.ipynb
```

Seeds are fixed at 42. The notebook writes the trained model to `models/` (git-ignored).

## Repository layout

```
house_price_prediction.ipynb   full workflow with outputs and figures
docs/presentation.pptx         10-slide project presentation
data/README.md                 how to get the Kaggle data
requirements.txt
```

## Team

**Coffee Mates**: Sacharith Reddy Jambula and Manohar Naddunoori.
Deep Learning, Project 2. M.Sc. Big Data & AI, SRH University of Applied Sciences, Leipzig.
