## scheduled learning rates

Sheduling your learning rate can prove to be beneficial in a variety of forms, it can potentially let your model converge on a global minima with more precision, help your model avoid local minima and escape saddle points, and provide faster training.

Doing so is entirely dependent on other hyperparameters such as bounds of your learning rate, momentum parameter, variance & mean of your inputs, and others.

Scheduling a learning rate is typically done so by ***decaying*** the learning rate to a smaller value as the model continues to train further, to mitigate oscillations as the model begins to converge onto a global optima of the loss space.

Due to the nature of mini-batch gradient descent, which has more frequent weight updates per epoch, larger magnitudes of oscillations in the search space of the loss function might be present, which can be useful in the earlier stages of training a model, but in the latter stages when the model is beginning to converge on an optima, these oscillations can make way for falliblity in the model predictions.

Also, especially when using vanilla gradient descent, without a modified optimizer with momentum, RMSprop, Adam and it's variants, weight updates might continue to oscillate further making it difficult for a model to reach it's optima.

***Decaying*** the learning rate, allows for a model to take smaller, careful, yet more precise steps as it begins to converge on the optimum set of parameters to minimize the loss.

### halving

Halving, is a pretty simple and straightforward means to adjust the learning rate at a given iteration or epoch.

Halving is implemented by a simple division or multplication by $.5$, after a predefined number of epochs, accuracy value, or loss value is reached

```
def gradient_descent(x, y, ... epochs)
    for epoch in range(epochs)
        '''
        Include forward, backward, & weight update.
        '''

        if epoch > 400:
            alpha *= .5
```

It's pretty straightforward and simple but when implemented at the right time, typically when the model begins to near the point where it needs precision to reach the global minima, it can significantly improve the optimization process and increase training effectiveness.

Of course, the 'halving' isn't limited to halving the learning rate, you can also adjust the learning rate over time, by a smaller fraction, though it's entirely dependent on what can help a model train better.

### exponential decay

Scheduling a learning rate based on exponential decay depends on a hyperparamter, $k$, which is known as the decay rate and the current time step, $t$, which is the total number of weight updates / iterations a model has gone over thus far.

<div align = 'center'>

$\alpha = \alpha * e^{-kt}$

</div>

This equation them smoothly, exponetially decays the learning rate, $\alpha$, as the number of iterations, $t$ increases over time.

<div align = 'center'>
<img src = '../util_images/expdecay.png' width = 400></img><br>
<span style = 'font-size: 12px'> An exponentially decaying learning rate </span>
</div>

## cyclical learning rates, 1cycle policy, & superconvergence

A cyclical learning rate is a learning rate schedule that involves carying the learning rate between a range of minimum or maximum bounds.

These minimum and maximum bounds are typically found through conjectures of what the optimum learning rate could be, and then empirical testing over time.

A common type of cyclical learning rate policy is known as the *triangular learning rate policy* where the learning rate varies cyclically on a triaingular path, meaning lienarly increasing to the maximum bound and then linearly decreasing to the minimum bound over a set number of epochs, which is known as the cycle size.




### superconvergence
