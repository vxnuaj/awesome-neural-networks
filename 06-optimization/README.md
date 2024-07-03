## optimization

In addition to traditional gradient descent, which involves:

<div align = 'center'>

$Z1 = forward(W1, X, B1)$

$A1 = ReLU(Z1)$

$Z2 = forward(W1, X, B2)$

$A2 = Softmax(Z2)$

$∂W1, ∂B1, ∂W2, ∂B2 = Backward(X, Y_{onthot}, W2, A2, A1, Z1)$

$W1, B1, W2, B2 = Update(W1, B1, W2, B2, ∂W1, ∂B1, ∂W2, ∂B2, \alpha)$

</div>

There are other means, better algoithms, that can be implemented to modify the traditional gradient descent, and make way for faster training and convergence of a neural network.

> [!NOTE]
> *Of course, at least when used right.*

Some of these improved optimizers, include momentum, RMSprop, Adam, and AdaMax, all involving a foundational concept called [Exponentially Weighted Averages](https://en.wikipedia.org/wiki/Exponential_smoothing).

### exponentially weighted averages

> [EWA Sample Code](EWA.py)

Exponentially weighted averages (EWA), or known as exponential moving averages, are a means to average and smooth out a set of data points over time $t$, based on a smoothing parameter, $\beta$.

The difference between an EWA and a simple moving average (SMA), lies in the fact that the EWA puts more weight on the more recent data points, while the SMA puts equal weight on all data points through a period of time. 

Therefore, if a set of datapoints appear to be in an upward trend, the EWA at time $t$, will be at a greater value than the SMA, as the SMA, doesn't consider the recent upward trend to be more important than previous datapoints.

Inversely, the EWA at a time $t$, when a set of datapoints is in a downward trend,will be at a lower value when compared to a SMA, as the SMA doesn't consider the downward trend to be of higher importance in comparison to past data points. For the SMA, everything is equivalent

<div align = 'center'>
<img src = "../util_images/averages.jpg" width = 600></img><br>
<em style = 'font-size:12px'> Notice the difference, between the EWA and SMA</em>
</div>
<br>

When comparing the EWA to a traditional weighted moving average (WMA), the difference lies in how the weights of both decay. The EWA decays it's weights over $t$ exponentially while the WMA decays it's weights over $t$ linearly.

> *This doesn't neccesarily mean that the smoothing of an EWA will be steeper than the WMA, due to the exponentially decaying nature of it's weights, that's all dependent on the smoothing parameter, β*.

The equation for computing an EWA is defined as:

<div align = 'center'>

$V\theta = \beta * V\theta_{t-1} + (1 - \beta) * \theta_t$

</div>

where $\beta$ is the smoothing parameter, $V\theta_{t-1}$ is the exponentially avereaged data point at the previous time step $t$, and $∂\theta_t$ is the derivative of the current data point at time $t$.

Now, while that's the meat of the formula for computing an exponentially weighted average, initially when doing so, there can be a type of bias that inhibits the EWA from truly averaging a set of datapoints at the beginning.

This is as the EWA is initially biased towards $0$ as in the above formula for $V\theta$, the initial input of $V\theta_{t-1}$, is at a value of 0 due to a lack of previous time steps, $t$.

<div align = 'center'>
<img src = '../util_images/biasedewa.png'></img>
</div>

An EWA with a bias would end up looking like the green line above, while an EWA without a bias would look like the red line.

To ensure the calculation of an EWA without a bias, we can then divide the output, $V\theta_t$, by a bias correction term which is defined as $1 - \beta^t$.

Then, the full equation becomes:

<div align = 'center'>

$V\theta_t = \frac{\beta * V\theta_{t-1} + (1 - \beta) * V}{1 - \beta^t}$
</div>

Now, the $\beta$ parameter determines how smooth the final output of averaging youdataset will be.

The higher a $\beta$ is, the more smoothed out your dataset will be when applying EWAs, while with the inverse, when your $\beta$ is a smaller value, the more aligned the final output of the EWA will be with your original dataset.

As an example, say I had a dataset of 50 samples, and to begin, I applied an EWA with a $\beta$ value of .7. 

The results, would look something like this:

<div align = 'center'>
<img src = '../util_images/ewa1.png' width = 550></img><br>
<em style = 'font-size:12px'> Where the orange line is the original data and the blue line is the exponentially weighted averaged data.</em>
</div>
<br>

Where it's clear that the exponentially averaged data points are more smoothed out than the original jagged dataset.

Say I apply a $\beta$ parameter with a value of $.3$, the difference would look like this:

<div align = 'center'>
<img src = '../util_images/ewa2.png' width = 550></img><br>
</div>

Here, the averaged dataset is more in line to the original data, with little modification to it.

Now, if I had a  $\beta$ value of 0:

<div align = 'center'>
<img src = '../util_images/ewa3.png' width = 550> </img><br>
</div>

Here, the averaged datapoints seems to be non-existent, as it's completely in line with the original dataset, meaning a $\beta$ of 0 has no effect.

Ultimately, EWAs are just a means of averaging a dataset overtime $t$,  in a manner that puts less weight on earlier data points than more recent data points in an exponentialy decaying manner.

### gradient descent with momentum

> *[Implementation of a Neural Network with momentum based gradient descent.](MomentumNN.py)*

Momentum based gradient descent, is a means to increase the speed of convergence of a neural network by adding a *momentum* which is able to give the model a slight "push" in the right direction towards the global optima.

The problem posed in the traditional gradient descent algorithm, was that it enables varying degrees of vertical oscillations in the learning path, ultimately decreasing the speed of convergence as a model isn't as intentional towards finding the local optima.

The cause of this being that the gradients of the loss with respect to a parameter can wildly vary and may not be smoothed throughout the process of training a model. 

<div align = 'center'>
<img src = '../util_images/vosci.png'></img><br>
<em style = 'font-size:12px'> For example, a learning path of gradient descent without momentum. </em>
</div>
<br>

While one could increase the learning rate, $\alpha$, it might be likely that an increased $\alpha$ would purely continue to increase the magnitude of the oscillations rather than set the learning path on a more direct and horizontal journey to the global optima.

While decreasing $\alpha$ might smooth out these oscillations, the consequence of doing so is a decreased step size and overall slower learning which can be negatively consequential for making the most out of a training run in a shorter period of compute time.

The ideal scenario would be to smooth these vertical oscillations, while retaining the value of $\alpha$, to optimize for faster and more direct learning to reach the global optima.

Essentially, this is what an exponentially weighted average (EWA) can make way for, by computing the averaged data points up until time point $t$.

If you recall how EWA smoothes data points, 

<div align = 'center'>
<img src = '../util_images/ewa1.png' width = 550></img>
</div>

the same process can be applied to the gradients of the loss with respect to a paramter, $\theta$, to smooth them out by averaging, allowing for less vertical oscillations within them, and then less vertical oscillations in the overall learning curve.

Ultimately then, when applied, the learning curve would end up being smoother, allowing for faster learning, while retaining a similar magnitude of $\alpha$.

<div align = 'center'>
<img src = '../util_images/momentum.png' width = 550></img>
</div>

> [!NOTE]
> *Notice how momentum based gradient descent is more intentional and is able to take larger step sizes when compared to vanilla gradient descent*

So, momentum with gradient descent, can be applied in a similar way to EWAs, we now average the gradients up to a current iteration step, $t$, and then use the average gradient within the weight update.

<div align = 'center'>

$∂W_1, ∂B_1, ∂W_2, ∂B_2 = backward(X, Y_{onehot}, W_2, A_2, A_1, Z_1)$

$V∂W_1 = \frac{\beta * V∂W_{1t-1} + (1 - \beta) * ∂W_1}{1 - \beta^2}$

$V∂B_1 = \frac{\beta * V∂B_{1t-1} + (1 - \beta) * ∂B_1}{1 - \beta^2}$

$V∂W_2 = \frac{\beta * V∂W_{2t-1} + (1 - \beta) * ∂W_2}{1 - \beta^2}$

$V∂B_2 = \frac{\beta * V∂B_{2t-1} + (1 - \beta) * ∂B_2}{1 - \beta^2}$

$W_1, B_1, W_2, B_2 = update(W_1, B_1, W_2, B_2, ∂VW_1, ∂VB_1, ∂VW_2, ∂VB_2, \alpha)$
</div>

Note, that in this case, we don't need to apply a smoothing value if note desired, in practice it can be ignored, with the consequence being that it might take a couple of iterations or epochs, depending on the depth of your neural network, for the averaged gradients to *"warm-up"* and represent a true moving average.

So then if desired, we can just define the averaged gradients for a parameter $\theta$, as:

<div align = 'center'>

$V∂\theta = \beta * (V∂\theta_{t - 1}) + ( 1 - \beta ) * ∂\theta$
</div>

without a smoothing term. 

In practice, when you set the $\beta$ term, it's typically between the value of $.9$ and $.999$, depending on the needs for your neural networks.

>[!NOTE]
> *If you're curious, check out an implementation [here](MomentumNN.py)*

### RMSprop

> *[Implementation of a Neural Network with RMSpropagation](RMSpropNN.py)*

Another algorithm, as a modified alternative gradient descent is Root Mean Squared Propagation or RMSprop for short.

RMSprop involves the decay of the learning rate, $\alpha$, as the gradients of the loss with respect to a parameter $\theta$, increase. 

Inversely, as a gradient, $\frac{∂L}{∂\theta}$, begins to decrease, the learning rate begins to increase.

Unlike traditional scheduling of $\alpha$, $\alpha$ is instead scaled over time $t$ by dividing $\alpha$ over the root mean squared (RMS) of the exponentially weighted averaged (EWA) gradients squared, $S∂\theta$.

So the equation to caluclate the EWA of $\theta^2$ is defined as:

<div align ='center'> 

$S∂\theta = \beta * S∂\theta_{t-1} + (1 - \beta) * ∂\theta^2$
</div>

Unlike previously, as done in momentum, when computing the exponentially weighted averages of the gradients, RMSprop typically doesn't incorporate bias correction in practice, as it's absence doesn't affect the algorithm as heavily, but in cases where the bias does inhibit learning, the bias correction value may be needed.

<div align = 'center'>

$S∂\theta = \frac{S∂\theta}{1 - beta^t}$<br>
<em style = 'font-size: 12px'> Which then, the bias correction woudl be applied like this. </em>
</div>

Thereafter, once $S∂\theta$ is calculated, you adaptively scale the learning rate, within the update rule ($\theta = \theta - \alpha * \frac{∂L}{∂\theta}$) through a division by the RMS of $S∂\theta$.

<div align = 'center'>

$\theta = \theta - \frac{\alpha}{\sqrt{S∂\theta^2 + \epsilon}} * \frac{∂L}{∂\theta}$<br>
<em style = 'font-size:12px'> Note the addition of a small epsilon to avoid division by 0. It's typically on the order of 1e-6 or smaller.</em>
</div>

Keep in mind that the exponentially averaged gradients, squared, is a fluctuating term over time. Increasing as the raw gradients increase, and decreasing as the raw gradients decrease.

Therefore, as $∂\theta$ increases, $S∂\theta$ will increase, then $\sqrt{S∂\theta^2}$ will increase, therefore the division of $\alpha$ over $\sqrt{S∂\theta^2}$ will result in a smaller alpha as $∂\theta$ increases and inversely as $∂\theta$ decreases. 

It's also important to note that RMSprop is more sensitive to the magnitude of a learning rate. While typically you might've been able to set a learning rate to $.1$ for instance, you'd need to scale you learning rate down to say $.01$ or $.001$ to make sure you don't introduce unwanted instability to your model.

So ultimately, if we put this all together for a model with parameters $W_1$, $B_1$, $W_2$, and $B_2$, it'd look as:

<div align = 'center'>

$∂W_1, ∂B_1, ∂W_2, ∂B_2 = backward(X, Y_{onehot}, W_2, A_2, A_1, Z_1)$

$S∂W_1 = \beta * S∂W_{1t-1} + (1 - \beta) * ∂W_1^2$

$S∂B_1 = \beta * S∂B_{1t-1} + (1 - \beta) * ∂B_1^2$

$S∂W_2 = \beta * S∂W_{2t-1} + (1 - \beta) * ∂W_2^2$

$S∂B_2 = \beta * S∂B_{2t-1} + (1 - \beta) * ∂B_2^2$

$W_1 = W_1 - \frac{\alpha}{\sqrt{S∂W_1^2 + \epsilon}} * \frac{∂L}{∂W_1}$

$B_1 = B_1 - \frac{\alpha}{\sqrt{S∂B_1^2 + \epsilon}} * \frac{∂L}{∂B_1}$

$W_2 = W_2 - \frac{\alpha}{\sqrt{S∂W_2^2 + \epsilon}} * \frac{∂L}{∂W_2}$

$B_2 = B_2 - \frac{\alpha}{\sqrt{S∂B_2^2 + \epsilon}} * \frac{∂L}{∂B_2}$

</div>

Similar to momentum, this may reduce the vertical oscillations in the learning path, but at the trade-off of having a smaller learning rate.

While in some situations, this can be beneficial as you'd want a smaller learning rate, if used improperly by tuning to the wrong $\beta$ value, a the adaptive learning rate may end up slowing you down.

It's also important to note, RMSprop doesn't smooth your gradients like a momentum term does, it only adaptively scales the learning rate. 

Therefore, your gradients might still be oscillating wildly in the vertical direction, but their impact onto the learning path would be mitigated by the smaller learning rate when the magnitude of their oscillations is larger.

Then in some cases, it might be beneficial to implement a combination of both, RMSprop and Momentum based gradient descent.

Just like momentum, when you set the $\beta$ term, it's typically between the value of $.9$ and $.999$, and again, depends on the needs for your neural networks.

>[!NOTE]
> *If you're curious, check out an implementation [here](RMSpropNN.py)*

### adaptive moment estimation

> *[Implementation of a Neural Network with adaptive moment estimation](AdamNN.py)*

Adaptive moment estimation, also known as Adam, is another variation of gradient descent, to optimize for faster learning, this time making use of the properties that are present in both RMSprop and Gradient Descent with Momentum.

Within the algorithm, the properties derived from the momentum based gradient descent and RMSprop are called the ***first order moment***, which is just the exponentially weighted average (EWA) of the gradients, and the ***second order moment*** which is the squared EWA of the gradients. 

The ***first order moment*** is simply the EWA of the gradients up until time point $t$.

<div align = 'center'>

$V∂\theta_t = \beta_1 * V∂\theta_{t-1} + ( 1 - \beta_1)* ∂\theta_t$<br>
<em style = 'font-size: 12px'>The same calculation as the original momentum term</em>
</div>

The ***second order moment*** represents the EWA of the squared gradients, more known as the uncentered variance, up until time point $t$.

<div align = 'center'>

$S∂\theta_t = \beta_2 * S∂\theta_{t-1} + ( 1 - \beta_2 ) * ∂\theta_t$<br>
<em style = 'font-size: 12px'> The same calculation as is done in RMSprop</em>
</div>

The rationale behind the second moment being the uncentered variance is, variance is typically calculated as $\frac{\sum{(X - \mu)^2}}{n}$, but given that the EWA already computes a running average, without subtracting the $\mu^2$ from the squared datapoints, it's more referred to as the uncentered variance.

Of course, if we subtracted the square of the first moment, $V∂\theta^2$, from the second moment, $S∂\theta$, we'd get the variance, but we don't need this term in the context of adam. 

It's important to note that in Adam, you apply the bias correction term $1 - \beta$ to both, the first and second moments, to mitigate an initial bias towards $0$ in the initial steps. 

While Adam can work without bias correction, it comes at the price of taking a few epochs to fully 'wamrup' your learning rate.

> [!NOTE]
> *Read more [here](https://arc.net/l/quote/kzfpckpd)*

Those two computed terms, $V∂\theta_t$ and $S∂\theta_t$, are then used in the same manner as is done in RMSprop and momentum. 

The first moment, $V∂\theta_t$ is used as the gradient in the update rule.

The second moment, $S∂\theta_t$, is used as the term to adapt the learning rate through a division of $\frac{\alpha}{S∂\theta_t}$.

Ultimately together, once they are both computed, the update rule then looks as:

<div align = 'center'>

$\theta = \theta - \frac{\alpha}{\sqrt{S∂\theta_t^2 + \epsilon}} * V∂\theta_t$<br>
<em style = 'font-size: 12px'> Notice the small epsilon to avoid a division by 0</em>
</div>

So lets say we had a 2 layer neural network with parameters of $W_1, B_1, W_2, B_2$. 

Put everything together, it all looks as:

<div align = 'center'>

$∂W_1, ∂B_1, ∂W_2, ∂B_2 = backward(X, Y_{onehot}, W_2, A_2, A_1, Z_1)$

$V∂W_1 = \beta * V∂W_{1t-1} + (1 - \beta) * ∂W_1$

$V∂B_1 = \beta * V∂B_{1t-1} + (1 - \beta) * ∂B_1$

$V∂W_2 = \beta * V∂W_{2t-1} + (1 - \beta) * ∂W_2$

$V∂B_2 = \beta * V∂B_{2t-1} + (1 - \beta) * ∂B_2$

$S∂W_1 = \beta * S∂W_{1t-1} + (1 - \beta) * ∂W_1^2$

$S∂B_1 = \beta * S∂B_{1t-1} + (1 - \beta) * ∂B_1^2$

$S∂W_2 = \beta * S∂W_{2t-1} + (1 - \beta) * ∂W_2^2$

$S∂B_2 = \beta * S∂B_{2t-1} + (1 - \beta) * ∂B_2^2$

$W_1 = W_1 - \frac{\alpha}{\sqrt{S∂W_1^2 + \epsilon}} * V∂W_1$

$B_1 = B_1 - \frac{\alpha}{\sqrt{S∂B_1^2 + \epsilon}} * V∂B_1$

$W_2 = W_2 - \frac{\alpha}{\sqrt{S∂W_2^2 + \epsilon}} * V∂W_2$

$B_2 = B_2 - \frac{\alpha}{\sqrt{S∂B_2^2 + \epsilon}} * V∂B_2$
</div>

It might seem that Adam proves to be a more robust optimizer, as it incorporates both momentum and an adaptive learning rate, but it's use-case purely depends on the specific model being built and the problem it's trying to solve.

Note that, when you're tuning of setting the initial hyperparamters for Adam, $\beta_1$ and $\beta_2$, $\beta_1$ is typically set to an initial value of at least $.9$, to have high weighting for recent gradients, while $\beta_2$ is set to an initial value of $.99$ to have an optimal estimate of the variance.

Determining the final values for both, usually comes down to a matter of empirical testing.

> [!NOTE]
> *Checkout an implementation of a Neural Network with Adam optimization [here](AdamNN.py)!*

### adamax

So Adam, essentially makes use of an $L2$ norm to estimate the second moment, $V∂\theta$, as it computes the average of the squared gradients as: $\beta * S∂\theta_{t-1} + ( 1 - \beta ) * ∂\theta^2$, and then takes the $\sqrt{}$ during the weight decay as: $\frac{1}{\sqrt{S∂\theta}}$.

Then the learning rate, $\alpha$ is adapted inversely to this $L2$ norm of the gradients by the division of $\frac{\alpha}{\sqrt{S∂\theta}}.$

*The term, $\sqrt{S∂\theta}$ can be conceptually considered the $L2$ norm of the gradients, up until iteration, $t$.*

It was found, [in the original paper for Adam](https://arxiv.org/pdf/1412.6980), that this inverse $L2$ norm of the gradients, can be generalized to an $LP$ based norm, only if an equation is derived when the limit of $p$ is set to go to $\infty$.

<div align = 'center'>

<img src = '../util_images/adamax1.png'></img>
<img src = '../util_images/adamax2.png' width = 500></img>
<img src = '../util_images/adamax3.png' width = 300></img><br>

<em> Derivation, as presented in the original [Adam Paper](https://arxiv.org/pdf/1412.6980)</em>
</div>

Then, the equation for the second moment simply becomes a matter of taking the $max$, the greatest value, between $(\beta_2 \cdot S∂\theta, |∂\theta|)$.

<div align = 'center'>

$S∂\theta = (\beta_2 \cdot S∂\theta_{t-1}, |∂\theta|)$
</div>

which is then, as previous in Adam, used in the update rule to adaptively scale the learning rate as:

<div align = 'center'>

$\theta = \theta - \frac{\alpha}{\sqrt{S∂\theta + \epsilon}}(V∂\theta)$
</div>

Again, as previous, determining the hyperparameters for $\beta_1$ and $\beta_2$ are a matter of empirical testing.

As for choosing between AdaMax and Adam, it's commonly said that AdaMax, in theory, is better suited for training models that involve embeddings and sparse weight updates.

### nesterov momentum

> *See [here](NesterovNN.py) for an implementation!*

Nesterov momentum is a variant of Momentum, which serves a similar, yet upgraded means to compute the optima of the loss in a faster and more direct manner when compared to momentum.

You, in a sense, allow for the Gradient Descent to 'look ahead' and predict what the optimal next jump would be. This is done through a 'lookahead'.

This 'lookahead' is computed by **first**, computing the gradient at the current position using the previously accumulated gradient and making a big jump in the direction of the previously accumulated gradient.

<div align = 'center'>
<img src = '../util_images/nesterov.png' width = 600>
</div><br>

This 'lookahead' term is more of a placeholder term, rather than serving as a real update of $\theta$

**2nd**, is then measuring the gradient at the location of where we ended up at after the big jump and then computing the accumulated gradient, to then finally compute the true update of $\theta$ using the newly accumulated gradient.

Then this process is repeated for all time steps / iterations.

This can be defined as:

<div align = 'center'>

$\theta_{lookahead} = \theta - \beta* v\theta_t$ 

Compute: $∂J(\theta_{lookahead})$

$v\theta_t = \beta * v\theta_{t-1} + ( 1 - \beta ) * ∂J(\theta_{lookahead})$

$\theta = \theta - \alpha * v\theta_t$
</div>

This then allows for the model to conjecture where the optimal jump might be and then correct after making that jump.

Essentially, the $∂J(\theta_{lookahead})$ is added onto the $\beta * v\theta_{t-1}$, as a means of 'correcting' the error that would've been made from purely relying on the past accumulated gradients.

While in regular momentum, the big jump would be made without any additional correction prior to the next iteration. The jump or weight update, would've just been made based on the current gradient and the accumulated past gradients without any intermediate error-correction.

### nadam

> *Checkout the implementation of Nadam [here](NadamNN.py)!*

Nadam, merges the Adam optimizer and the Nesterov Momentum, by replacing the original computation of Momentum with Nesterov Momentum.

This then allows for Adam to leverage the benefits that Nesterov Momentum may have in finding a more precise gradient while still making use of the second moment term (RMSprop)

**So, mathematically, this can be defined as:**

<div align = 'center'>

$\theta_{lookahead} = \theta - \beta *  v\theta_{t-1}$

$a1, a2, z1, z2 = forward()$

$∂J(\theta_{lookahead}) = backward(a1, a2, z1, z2)$

$v\theta_t = \beta * v\theta_{t-1} + ( 1 - \beta) * ∂J(\theta_{lookahead})$

$s\theta_t = \beta * v\theta_{t-1} + (1 - \beta) * ∂J(\theta_{lookahead})^2$

$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{s\theta_t}} * v\theta_t$
</div>

The added Nesterov Momentum as the first moment, allows for the momentum term to make more precise predictions of the direction of the optimal gradient step, at a better manner than Momentum.
