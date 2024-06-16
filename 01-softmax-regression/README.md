# softmax regression

If you're familiar with [linear regression](https://www.youtube.com/watch?v=7ArmBVF2dCs), a model where you aim to construct a line of best fit to a set of linear data points ($\vec{X}$) through the equation, $y = WX + B$, and by iteratively adjusting the model parameters ($\vec{W}$ and $\vec{B}$), it's important to **not**, I repeat, **not,** get Softmax Regression confused with fitting a line of to a set of datapoints.


<p align="center">
  <img src="../util_images/linearreg.gif" width="400" align="center"><br>
  <span style="font-size:12px;">In essence, this is what linear regression aims to do.</span>
</p>


Softmax regression is not at all akin to linear regression in terms of outcome.

Rather, for intuition, you can refer to [logistic regression](https://www.youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe), where the outcome of a model isn't exactly to construct a line of best fit to a set of datapoints but rather to perform [binary classification](https://en.wikipedia.org/wiki/Binary_classification) on a given set of datapoints.

### a recap on logistic regression

In essence, logistic regression involves a combination of an affine transformation and the [`logistic`](https://en.wikipedia.org/wiki/Logistic_function) activation function, which is commonly called *sigmoid* and referenced as $\sigma()$, to perform [binary classification](https://en.wikipedia.org/wiki/Binary_classification).

**Let's touch on the logistic function first**

The `logistic` activation function can be mathematically defined as:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

It's shape takes on a continuous and smooth s-shaped curve, bounded at it's range between $[0, 1]$ and in it's domain, $[-\infty, \infty]$.

<p align = 'center'>
    <img src = "../util_images/logisticcurve.svg" width = 350><br>
    <span style = "font-size:12px">The s-shaped logistic curve</span>
</p>

Within this function, as the denominator grows ever-large, it's final output diminishes to near $0$ while inversely, as the denominator becomes increasingly small, the output converges to a near perfect $1$.

As an example of a near $0$ output, let's take an input $x$ that's extremely negative (at least "extreme" in terms of $\sigma$), for example $-20$:

$$
\sigma(-20) = \frac{1}{1 + e^{- (-20)}} \\~\newline
\sigma(-20) = \frac{1}{1 + e^{20}} \\~\\
\sigma(-20) = \frac{1}{1 + 485165195.40978} \\~\\
\sigma(-20) = \frac{1}{485165196.40978}\\~\\
\sigma(-20) = 2.06115362\times10^{-9} \approx .00000000206115
$$

As you can see, large negative values can lead to a near vanishing output of $\sigma$.

> *This will be an issue that we'll cover later*

Now for an example of an output that's near $1$, we can take an input $x$ that's positively large, for example, $20$:

$$
\sigma(20) = \frac{1}{1 + e^{- (20)}} \\~\\
\sigma(20) = \frac{1}{1 + e^{20}}\\~\\
\sigma(20) = \frac{1}{1 + 2.718281828459^{-20}} \\~\\
\sigma(20) = \frac{1}{1.0000000020612} \\~\\
\sigma(20) = .99999999793 \\~\\
$$

And if we want to reach the intermediary value, $.5$, we can define the input as $0$:

$$
\sigma(0) = \frac{1}{1 + e^{- (0)}} \\~\\
\sigma(0) = \frac{1}{1 + 1} \\~\\
\sigma(0) = \frac{1}{2} = .5\\~\\
$$

Pretty simple.