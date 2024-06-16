# softmax regression

If you're familiar with [linear regression](https://www.youtube.com/watch?v=7ArmBVF2dCs), a model where you aim to construct a line of best fit to a set of linear data points by iteratively adjusting the model parameters ($W$ and $B$), it's important to **not**, I repeat, **not,** get Softmax Regression confused with fitting a line of to a set of datapoints.

Softmax regression is not at all akin to linear regression.

Rather, for intuition, you can refer to [logistic regression](https://www.youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe), where the outcome of a model isn't exactly to construct a line of best fit to a set of datapoints but rather to perform [binary classification](https://en.wikipedia.org/wiki/Binary_classification) on a given set of datapoints.

### a recap on logistic regression

In essence, logistic regression involves the use of the [`logistic`](https://en.wikipedia.org/wiki/Logistic_function) activation function, which is commonly called *sigmoid* and referenced as $\sigma()$, to perform [binary classification](https://en.wikipedia.org/wiki/Binary_classification).

Now, the `logistic` activation function can be mathematically defined as:

$$
\sigma = \frac{1}{1 + e^{-x}}
$$

and it's shape takes on a continuous and smooth s-shaped curve, bounded at it's range at $[0, 1]$ and in it's domain, $[-\infty, \infty]$.