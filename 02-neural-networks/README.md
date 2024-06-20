## into neural nets

> *[Vanilla Neural Network Code](nn.py) | [Mini-Batch Neural Network Code](MiniBatchNN.py)*

> [!NOTE]
> *The foundations of the [logistic & softmax regression](../01-logistic-&-softmax-regression/README.md) serve as similar foundations for the architectures of neural networks, the difference being that neural networks are **deep** and **complex**.*

### the architecture

Revisiting the model of an artificial neuron:

<div align = 'center'>
<img src = '../util_images/neuron2.png' width = 450></img>
</div>

It can be recalled that a neuron takes in a set of input features, in the image above defined as $f_i$. 

In our case, we'll be defining these inputs as $X_i$, $i$ denoting the index of the feature.

The neuron takes in the set of inputs, $X_i$, applies the affine transformation $WX + B$ and then applies an activation function, either $\frac{1}{1 + e^{-z}}$ (sigmoid) or $\frac{e^{z}}{\sum{e^z}}$ (softmax), depending if you're performing binary classification or multi-class classification, to then output a final value which we'll denote as $\hat{Y}$.

A ***neural network*** has a similar flow in it's forward pass, but just as mentioned earlier, it's a lot more ***deep*** and ***complex***.

What is meant by ***deep*** and ***complex*** is, it contains more ***neurons*** and more ***layers***.

Here's a diagram for reference:

<div align = 'center'>
<img align = 'center' src = '../util_images/nn.png' width = 450></img><br><br>
<span style = 'font-size:13px'>Note: Most definitely, <b>not to scale</b></span>
</div>

<br>

Discussing the layers, the left-most layer of this neural network is simply called the ***input layer***. 

This input layer isn't really a layer of the network itself, it's rather a representation of a set of input features to the ***first hidden layer***, where each node is representative of a single input feature.

> [!NOTE]
> *A single input feature, for example, can be a single pixel value of an image*

### the forward pass

Now, every single node / input feature within the input layer is fed into every single neuron individually in the first hidden layer.

> [!NOTE]
> *Typically, the input layer isn't considered as part of the total layer count nor is even called a layer at times. So here, when a "first layer" is mentioned, we'll be referring to the first hidden layer of a neural network, not the input layer*

In prior single neuron examples, an affine transformation was computed to then be fed into an activation function to get a final output.

When we feed an input feature into a neuron in the hidden layer of the network, the same exact process occurs, but per neuron.

In our case though, we'll be replacing the $\sigma$ activation function for a $ReLU$ activation. 

The rationale behind this being that $\sigma$ can prove to be unstable for deep neural networks, given that they're prone to vanishing gradients, and are more expensive to compute given $e$.

### TODO
- [ ] Add link to activation functions page to serve as an explanation, whenever ReLU is ment ioned

> [!NOTE]
> *For the following, it'll be assumed that all inputs, to both the hidden layer and output layer are **vectorized** outputs of a previous layer denoted by a capital variable.*
> 
> *In this case, the inputs are then matrices of dimensions - $(n_{in}, samples)$- where $n_{in}$ are the number of input features to a given layer*.
>
> *To learn more about vectorization, check out this amazing [resource](https://youtu.be/qsIrQi0fzbY).*

<div = align= 'center'>

$for$&nbsp;$each$&nbsp;$neuron$,&nbsp;$n$,&nbsp;$in$&nbsp;$hidden$&nbsp;$layer:$

$z_1^n = w_1^nX + b_1^n$

$a_1^n = ReLU{(z_1^n})$<br><br>
<span style = 'font-size:12px'>The subscript $1$, denoting the first hidden layer.</span>
</div>

Then, the outputs of the hidden layer, $A_1$ in vectorized form for all neurons $n$, are fed into the output layer where the same process, an affine transformation and a non-linear activation, in this case will be softmax ($\tilde{\sigma}$), take place to allow for a final output.

<div align = 'center'>

$for$&nbsp;$each$&nbsp;$neuron$,&nbsp;$n$,&nbsp;$in$&nbsp;$output$&nbsp;$layer:$

$z_2^n = w_2^nA_1 + b_2^n$

$\hat{Y} = a_2^n = \tilde{\sigma}(z_2^n)$

</div>

It's important to note, given that we're using the softmax function, the final output is represented in terms of probability.

Just as prior in softmax regression, where we take take the $argmax$ of the final output vector, to get a final class prediction, we can do the same here.

<div align = 'center'>

$pred = argmax(A_2)$

</div>

This value $pred$, given that your labels, $Y$, are encoded into integer representations, can be used to compute an accuracy by averaging the amount of times where $pred = Y$ is true over the number of samples in your dataset

<div align = 'center'>

$accuracy = \frac{\sum{pred == Y}}{samples}$ <br><br>
<span style = 'font-size: 12px'>Pseudo code for computing accuracy, where $\hat{Y} == Y$ would return a boolean value.</span>
</div>

> [!IMPORTANT]
> *When training a neural network, you typically wouldn't use the $argmax$ed values to computed the loss or to compute gradients to train the model. You'd want to use the raw outputs, $A_2$ as a means to calculate the loss and the gradients as it's more representative of the true outputs of the neural network.*

So now, given the above, we can define a full forward pass of a neural network as:

> [!NOTE]
> ***Note** that from now, all inputs and output values, and paramters will be expressed in [vectorized](https://youtu.be/qsIrQi0fzbY) formats.*

<div align = 'center'>

$Z_1 = W_1X + B_1$ 

$A_1 = ReLU(Z_1)$

$Z_2 = W_2A_1 + B_2$

$A_2 = \tilde{\sigma}(Z_2)$

</div>

Now again, as prior, we can compute the loss using the, ***categorical cross entropy loss function***, just as prior in softmax regression.

<div align = 'center'>

$L(\hat{Y}, Y) = Y_{onehot} * ln(\hat{Y})$<br><br>
<span style = 'font-size: 12px'>Again, where $Y_{onehot}$ are the one-hot encoded labels.</span>
</div>

### the backpropagation

> [!NOTE]
> *We'll be going pretty heavy into calculus and linear algebra here. It's
> important to understand the mathematical foundations, at least that is if you
> truly want to become knowledgeable in deep learning.*
>
>*If you know the first principles, you'll know how to build creatively from them,
> to build novel ideas.*
>
> *Andrej Karparthy put out a more thorough rationale of why you should become a "backprop ninja", read it [here](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)*

Just as prior, backpropagation involes the calculation of the gradients of the loss, $L(\hat{Y}, Y)$, with respect to the given parameters, in this case being $W_1$, $B_1$, $W_2$, and $B_2$.

> [!NOTE]
> *I'll be interchangeably using $\theta_l$ to denote either parameters at the $l$th layer*

And again, just as prior, this involves the use of the chain rule of calculus.

To compute the gradients with respect to parameters in layer 2, $\frac{∂L(\hat{Y}, Y)}{∂\theta_2}$, and with respect to parameters in layer 1, $\frac{L(\hat{Y}{Y})}{∂\theta_1}$, we'll have to dig all the way back through the chain rule by computing the gradients of earlier variables.

For the parameters of the second layer, this will look as:

<div align = 'center'>

$\frac{L(\hat{Y}, Y)}{\theta_2} = (\frac{L(\hat{Y}, Y)}{∂A_2})(\frac{∂A_2}{∂Z_2})(\frac{∂Z_2}{∂\theta_2})$

</div>

For the parameters of the first layer, this will look as:

<div align = 'center'>

$\frac{L(\hat{Y}, Y)}{\theta_1} = (\frac{L(\hat{Y}, Y)}{∂A_2})(\frac{∂A_2}{∂Z_2})(\frac{∂Z_2}{∂A_1})(\frac{∂A_1}{∂Z_1})(\frac{∂Z_1}{∂\theta_1})$
</div>

This might seem complicated at first, but can all be dumbed down into simpler derivations, which can be understood with basic knowledge of calculus..

When computing, $(\frac{∂L(\hat{Y}, Y)}{∂A_2})(\frac{∂A_2}{∂Z_2})(\frac{∂Z_2}{∂\theta_2})$, we'd need to find $\frac{∂L(\hat{Y},{Y})}{∂Z_2}$ prior.

Now $\frac{∂L(\hat{Y},{Y})}{∂Z_2}$, can be simplified to $A_2 - Y_{onehot}$
> [!NOTE]
> *To keep things simple, I won't be going over this derivation here.*

Given this, what's left is finding the value of $\frac{∂Z_2}{∂\theta_2}$, which differs depending on the parameter we're regarding to.

Let's say we try to calculate the gradient with respect to $W_2$.

Given the original equation in the forward pass:

<div align = 'center'>

$Z_2 = W_2A_1 + B_2$

</div>

the gradient, $\frac{∂Z_2}{∂W_2}$ ends up being equal to $A_1$, given that the derivative of $W_2$ is $1$ and $B_2$ cancels out as it's a constant in reference to the gradient.

> [!NOTE]
> *If you've previously learnt calculus, this might come off as fairly easy, which it can be at times.* 

So ultimately our equation for $\frac{∂L(\hat{Y}, Y)}{∂W_2}$ will look like:

<div align = 'center'>

$\frac{∂L(\hat{Y}, Y)}{∂W_2} = (A_2 - Y_{onehot}) \cdot A_1^T$

</div>

To compute the gradient $\frac{\partial L}{\partial W_2}$ during the backward pass, the dimensions of the matrices must match for the multiplication: $(A_2 - Y_{onehot}) \cdot A_1^T$, which is why we take the transpose of $A_1$.


> [!NOTE]
> *This is where linear algebra might come in handy, read more on matrix multiplication and other common linear algebra operations [here](https://www.quantstart.com/articles/matrix-algebra-linear-algebra-for-deep-learning-part-2/).*

In practice, you'll want to average the gradient of each parameter over the total number of samples in each forward pass.

So given that we're already taking the matrix multiplication of $(A_2 - Y_{onehot})$ with $A_1^T$, all that's left is a division by the number of samples given that the matrix multiplication already includes an implicit $\sum$.

<div align = 'center'>

$\frac{L(\hat{Y}, Y)}{W_2} * \frac{1}{m}, m = sample$&nbsp;$size$

</div>


We can calculate $\frac{L(\hat{Y}, Y)}{∂B_2}$ in a similar manner.

Given the equation in the forward pass:

<div align = 'center'>

$Z_2 = W_2A_1 + B_2$

</div>

the gradient, $\frac{∂Z_2}{∂B_2}$ ends up being equal to $1$, given that the derivative of $B_2$ is $1$ and $W_2A_1$ cancels out as it's a constant in reference to the gradient of $B_2$.

Then  $\frac{L(\hat{Y}, Y)}{∂B_2}$ is equal to:

<div align = 'center'>

$\frac{L(\hat{Y}, Y)}{B_2} = (A_2 - Y_{onehot}) \cdot 1$

$\frac{L(\hat{Y}, Y)}{B_2} = (A_2 - Y_{onehot})$

</div>

Again in practice, we'd want to take the average the gradients of each parameter over the total number of samples in each forward pass.

In this case, given that we aren't computing a matrix multiplication, we can just apply a $\sum$ and divide by number of samples, $m$. Also note, that given the dimensions of the gradient, $\frac{L(\hat{Y}, Y)}{B_2}$, being $(n_{in}, samples)$, we'd want to sum over the second dimension of the matrix, $samples$ to properly average the gradients for each neuron.

If $m$ is defined as our total $samples$, this may look as:

> [!NOTE]
> *I'm combining the math, with NumPy's ability to specify the axis to sum over and whether we want to keep the dimensions when performing `np.sum`.*

<div align = 'center'>
<em>Pseudocode:</em> <br><br>

$(\frac{1}{m}) * \sum(\frac{L(\hat{Y}, Y)}{B_2}, axis = 1, keepdims = true)$

</div>

I'll be referring to $\frac{L(\hat{Y}, Y)}{W_2}$, $\frac{L(\hat{Y}, Y)}{B_2}$, and $\frac{L(\hat{Y}, Y)}{Z_2}$ as $∂W_2$, $∂B_2$, and $∂Z_2$ respectively to keep things simple

Now that the gradients of the loss with respect to the parameters in the outer layer are computed, we can get the gradients with respect to the parameters in the ***hidden layer***.

The gradients with respect to $W_1$ will look as:

<div align = 'center'>

$\frac{L(\hat{Y}, Y)}{W_1} = (\frac{L(\hat{Y}, Y)}{Z_2})(\frac{∂Z_2}{∂A_1})(\frac{∂A_1}{∂Z_1})(\frac{∂Z_1}{∂W_1})$

</div>

Given that we already know $∂Z_2$, we can simplify as:

<div align = 'center'>

$\frac{L(\hat{Y}, Y)}{W_1} = (A_2 - Y_{onehot})(\frac{∂Z_2}{∂A_1})(\frac{∂A_1}{∂Z_1})(\frac{∂Z_1}{∂W_1})$
</div>

Now, $\frac{∂Z_2}{∂A_1}$, given the original equation of the forward pass, $Z_2 = W_2A_1 + B_2$, can be simplified as follows:

<div align = 'center'>

$Z_2 = W_2A_1 + B_2$

$\frac{∂Z_2}{∂A_1} = W_2$

</div>


Then $\frac{∂A_1}{∂Z_1}$ can be calculated as the gradient of the $ReLU$ activation with respect to $Z_1$. 

<div align = 'center'>

$\frac{∂ReLU}{∂Z_1} = \begin{cases} 1, Z_1 > 0 \\ 0, Z_1 < 0\end{cases}$
</div>

We'll call this $∂ReLU(Z_1)$ to keep things simpler.

Up until now, we've essentially computed the needed components for $\frac{L(\hat{Y}, {Y})}{∂Z_1}$ or $∂Z_1$

<div align = 'center'>

$∂Z_1 = (W_2^T \cdot ∂Z_2) * ∂ReLU(Z_1)$

</div>

Finally, $\frac{∂Z_1}{∂W_1}$, given the equation for $Z_1$ in the forward pass, $Z_1 = W_1X + B_1$, can be calculated as follows:

<div align = 'center'>

$Z_1 = W_1X + B_1$

$\frac{∂Z_1}{∂W_1} = X$

</div>

So putting everything together, our final gradient, $\frac{L(\hat{Y}, Y)}{W_1}$, looks as:

<div align = 'center'>

$\frac{L(\hat{Y}, Y)}{W_1} = (\frac{L(\hat{Y}, Y)}{Z_2})(\frac{∂Z_2}{∂A_1})(\frac{∂A_1}{∂Z_1})(\frac{∂Z_1}{∂W_1})$

$\frac{L(\hat{Y}, Y)}{W_1} = ((W_2^T \cdot ∂Z_2) * ∂ReLU(Z_1)) \cdot X^T$

$∂W_1 = \frac{L(\hat{Y}, Y)}{W_1} = ∂Z_1 \cdot X^T$

</div>

Just as prior, we're transposing $X$ to ensure that it's dimensions are in alignment with $(W_2^T \cdot ∂Z_2) * ∂ReLU(Z_1)$ or $∂Z_1$ for the matrix multiplication.

Just as before, we need to average this gradient over the total number of samples in the forward pass. This can be done by purely dividing by the total number of samples, $m$, as the matrix multiplication involved an implicit $\sum$.

<div align = 'center'>

$∂W_1 = \frac{∂W_1}{m}$

</div>

Now we can compute the gradients of the loss with respect to $B_1$ in a very similar manner.

Note the gradient with respect to $∂B_1$ below.

<div align = 'center'>

$\frac{L(\hat{Y}, Y)}{B_1} = (\frac{L(\hat{Y}, Y)}{Z_2})(\frac{∂Z_2}{∂A_1})(\frac{∂A_1}{∂Z_1})(\frac{∂Z_1}{∂B_1})$

</div>

It's very similar to the above computation of $∂W_1$ with the difference being in that we calculate $\frac{∂Z_1}{∂B_1}$ instead of $\frac{∂Z_1}{∂W_1}$ at the end.

This can be computed as:

<div align = 'center'>

$Z_1 = W_1X + B_1$

$\frac{∂Z_1}{∂B_1} = 1$

</div>

So essentialy, $\frac{∂Z_1}{∂B_1}$ simplifies as:

<div align = 'center'>

$\frac{L(\hat{Y}, Y)}{B_1} = (\frac{L(\hat{Y}, Y)}{Z_2})(\frac{∂Z_2}{∂A_1})(\frac{∂A_1}{∂Z_1})(\frac{∂Z_1}{∂B_1})$

$∂B_1 = \frac{L(\hat{Y}, Y)}{B_1} = ∂Z_1 \cdot 1$
</div>

Again, we must average the gradient $∂B_1$ over the total number of samples, $m$, in our dataset, which can be done as:

> [!NOTE]
> *As before, I'm combining the math, with NumPy's ability to specify the axis to sum over and whether we want to keep the dimensions when performing `np.sum`.*

<div align = 'center'>

<em style = 'font-size: 14px'> Pseudocode</em><br><br>
$∂B_1 = \frac{1}{m} * \sum(∂B_1, axis = 1, keepdims = True)$
</div>

Again, we're doing so over $axis = 1$, as the first axis specifies the number of samples in our dataset.

Now putting this entire process together, the computation for the gradients looks as:

<div align = 'center'>

$∂Z_2 =  \frac{∂L(\hat{Y}, Y)}{∂Z_2} = (A_2 - Y_{onehot})$

$∂W_2 = \frac{1}{m} * ((\frac{∂L(\hat{Y}, Y)}{∂Z_2})(\frac{∂Z_2}{∂W_2})) = \frac{1}{m} * ((A_2 - Y_{onehot}) \cdot A_1^T)$

$∂B_2 = \frac{1}{m} * \sum((\frac{∂L(\hat{Y}, Y)}{∂Z_2})(\frac{∂Z_2}{∂W_2})) = \frac{1}{m} * \sum(A_2 - Y_{onehot}, axis = 1, keepdims = True)$

$∂Z_1 = (\frac{∂L(\hat{Y}, Y)}{∂Z_2})(\frac{∂Z_2}{∂A_1})(\frac{∂A_1}{∂Z_1}) = (W_2^T \cdot ∂Z_2) * ∂ReLU(Z_1)$

$∂W_1 = \frac{1}{m} * ((\frac{L(\hat{Y}, Y)}{Z_2})(\frac{∂Z_2}{∂A_1})(\frac{∂A_1}{∂Z_1})(\frac{∂Z_1}{∂W_1})) =  \frac{1}{m} * (((W_2^T \cdot ∂Z_2) * ∂ReLU(Z_1)) \cdot X^T)$

$∂B_1 = \frac{1}{m} * \sum((\frac{L(\hat{Y}, Y)}{Z_2})(\frac{∂Z_2}{∂A_1})(\frac{∂A_1}{∂Z_1})(\frac{∂Z_1}{∂B_1})) = \frac{1}{m} * \sum((W_2^T \cdot ∂Z_2) * ∂ReLU(Z_1), axis = 1, keepdims = True)$ <br>

<em>The more simpler derivations, being rightmost</em>
</div>

This might seem very verbose right now, but it'll look extremely simple in code. If you can fully understand this, you'll have absolutely no trouble implementing this in code because you'll have an exceptional understanding of the foundations.

An important thing to be wary of when training your model is the risk of introducing **vanishing** or **exploding** **gradients**.

While in shallow models, this might pose to be much of a problem, once you get into deep neural networks, the gradients at earlier layers might begin to explode or vanish, ultimately being unable to effectively learn.

Say we're using the $\sigma$ activation function. The maximum value for it's gradient is $.25$. Given that we use the chain rule to backpropagate the gradients onto earlier layers, this involves an increasingly lengthy multiplication the deeper a network gets.

So if the maximum gradient for $\sigma$ is $.25$ and say the gradietns for each layer did manage to be at $.25$ during the first training pass, if we use $\sigma$ for all of our hidden layers, in say a $10$ layer network, during back propagation the gradients for the first layer would be calculated as:
<br><br>
<div align = 'center'>

$∂W_1 = (.25)(.25)(.25)(.25)(.25)(.25)(.25)(.25)(.25)(.25)(.25)$

$∂W_1 = 0.00000095367431640625$

</div>

The gradient for $∂W_1$ turns out to be extremely small, which isn't ideal for training the earlier layers of the model. Combined with commonly smaller learning rates, this problem becomes worse.

Of course in practice, it isn't often that you'll see the use of $\sigma$, you'll more often see $ReLU$ or other variants being used as an activation function which mitigate the issue of vanishing gradients.


### weight update

Now, we can compute the weight update for any given parameter, $\theta_l$, using the same formula as prior:

<div align = 'center'>

$\theta = \theta - \alpha * ∂\theta$<br>
<span style = "font-size: 13px"> Where $\alpha$ is the learning rate</span>
</div>

So for each parameter, $W_1$, $B_1$, $W_2$, and $B_2$, the updates would look as:

<div align = 'center'>

$W_1 = W_1 - \alpha * ∂W_1$

$B_1 = B_1 - \alpha * ∂B_1$

$W_2 = W_2 - \alpha * ∂W_2$

$B_2 = B_2 - \alpha * ∂B_2$<br>
</div>

### gradient descent

Just as before, one forward and backward pass, meaning everything we've just computed, completes one pass of gradient descent.

So to put it all together, it looks as:

<div align = 'center'>

$for$&nbsp;$epoch$&nbsp;$in$&nbsp;$range(epochs):$

$Z_1 = W_1X + B_1$ 

$A_1 = ReLU(Z_1)$

$Z_2 = W_2A_1 + B_2$

$A_2 = \tilde{\sigma}(Z_2)$

$L(\hat{Y}, Y) = Y_{onehot} * ln(\hat{Y})$

$∂Z_2 =  (A_2 - Y_{onehot})$

$∂W_2  = \frac{1}{m} * ((A_2 - Y_{onehot}) \cdot A_1^T)$

$∂B_2 = \frac{1}{m} * \sum(A_2 - Y_{onehot}, axis = 1, keepdims = True)$

$∂Z_1 = (W_2^T \cdot ∂Z_2) * ∂ReLU(Z_1)$

$∂W_1 =  \frac{1}{m} * ((∂Z_1 * ∂ReLU(Z_1)) \cdot X^T)$

$∂B_1 = \frac{1}{m} * \sum(∂Z_1, axis = 1, keepdims = True)$

</div>

> [!IMPORTANT]
> *If you're curious to this process in code, through the implementation of a neural network, check it out [here](nn.py)!*

### mini-batch gradient descent

There's something called ***mini-batch gradient descent***, where a single pass of gradient descent and the respective training step, isn't representative of your entire dataset. 

Mini-Batch Gradient Descent is similar to Gradient Descentin the algorithmic sense, with the difference that it processes smaller batches of an entire dataset prior to taking a training step, rather than processing the entire dataset at once and then taking a training step.

So given that an ***epoch*** is an entire pass through your dataset, an model can take ***multiple training steps*** prior to finishing an epoch.

This can improve computation time and improve learning speed of a model, especially when there are a large number of training samples in your dataset.

For some intuition, say you have training set $X$.

$X = (784, 60000)$, where there are $784$ features and $60000$ samples.

We can split up $X$ into $6$ mini-batches of $10000$ samples each:

<div align = 'center'>

$X^{1} = (784, 10000)$

$X^{2} = (784, 10000)$

$X^{3} = (784, 10000)$

$X^{4} = (784, 10000)$

$X^{5} = (784, 10000)$

$X^{6} = (784, 10000)$
</div>

To then feed each $X^{t}$ in once and taking a training step in between each, then restarting from $X^{1}$ once more.

The trend of the loss function will be not be as smooth when compared to processing larger or the entire batch at once, due to the fact that within each forward pass, your model is operating on new unseen samples, but ultimately if done right, the value of the loss should still trend downwards.

Another benefit of processing data in mini-batches is the tiny bit of regularization it provides, to keep your model from overfitting on your dataset, though it isn't much.

Some principles to keep in mind when choosing a mini-batch size are:

- If you have a small training set, where samples, $m$, is $ < ~2000$ make use of full-batch gradient descent instead
- Typical mini-batch sizes are on orders of two, ${32, 64, 128, 256, etc}$, given how modern computer chips are built, process data like this is more optimal for ensuring you get the biggest efficiency.
- Make sure your mini-batch fits in your CPU / GPU nmemory
- If you're using BatchNorm ( which we'll go over later ), very small batches can lead to poor estimates of batch statistics ($\mu$ and $\beta$).

> [!IMPORTANT]
> *If you want to see a neural network, that implements mini-batch descent, check it out [here](MiniBatchNN.py)!*

## activation functions

> Checkout the implementations for each activation function in this [Jupyter Notebook](ActivationFuncs.ipynb)

So far, the discussed activation functions have been sigmoid ($\sigma$), softmax ($\tilde{\sigma}$), and $ReLU$. Other ones include $Leaky$&nbsp;$ReLU$, $tanh$, $PReLU$, $SeLU$, and many more. 

These all come with their unique properties that make some useful for specific scenarios and others not, which all depends on the type of model being built.

But each activation function should have a key set of needed features that allow for effective learning and efficient computation.

<details><summary> <b>Non-linearity</b> </summary>
The introduction of nonlinearity given by an activation function is extremely important for a neural network as it allows for it to capture complex features and datapoints from a set of inputs. 

Without activation functions, a neural network would essentially be a set of linear classifiers, unable to produce quality results to solve complex problems.</details>



<details><summary> <b>Range</b> </summary>
Activation functions have different ranges, some ranging from 0 to 1, others from 0 to infinity, and others from 1 to 1.

The output range of an activation function can be important to consider, for a variety of factors. 
	
It's range can affect the scale of gradients during backpropagation leading to exploding or vanishing gradients, the numerical stability, and the interpretability of an activation function (some, such as softmax, can be seen as a probability).</details>


<details><summary> <b>Monotonicity</b> </summary>Monotonicity, ensures that as the input of a model changes, the output remains consistently moving in a given direction (increasing or decreasing), which alongside it's smoothness is essential to a good loss function.

If a function isn't monotonic, the gradient sign may change irregularly over different inputs which can then lead to the model converging to the wrong set of parameters at a local minima instead of a global minima. This then makes the model less interpretable in how it learns.</details>

<details><summary> <b>Continuity</b> </summary>Continuity in an activation function allows for smooth continuous changes. 
    
As the inputs change, the outputs change slightly. This becomes important for ensuring smooth gradient descent computations during backpropagation. 
    
A lack of continuity, would imply that an activation function isn't differentiable at a given point, which may ultimately break the backpropagation and gradient descent.
</details>

	
<details><summary> <b>Differentiability</b> </summary>Differentiability allows for gradient based optimization algorithms (backpropagation) to properly allow gradients to be computed for the weight updates. 

If differentiability wasn't the case within the domain of an activation function, it's possible that NaN values begin to propagate throughout a model.</details>

<details><summary> <b>Sparsity</b> </summary>The sparsity of activation functions can be beneficial when looking for more efficient computations. The more sparse an activation function is, the less computations it might have to compute in order to reach it's output, then also leading to better memory efficiency.

Sparsity can also improve regularization. By encouraging neurons to be inactive at $0$, it reduces complexity which in certain contexts can mitigate overfitting and improve generalization.

Too much sparsity, on the other hand, can limit the power of a model to learn complex and nuanced patterns in a dataset. 

An activation function should be sparse enough to provide efficient computations and good regularization.</details>

<details><summary> <b>Computational Complexity</b> </summary>	The more computationally expensive an activation function is to compute, the slower a neural network will take to learn and then provide inference. 

It's better to use activation functions that are computationally efficient, in order to speed up the learning process and save valuable resources.</details>

> [!NOTE]
> *I'll only be covering the foundational activation functions, here. There's so much more out there, such as [GeLU](https://paperswithcode.com/method/gelu), [SeLU](https://paperswithcode.com/method/selu), and [more](https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons)*.

### σ

> [Code](ActivationFuncs.ipynb#cell-2)

The sigmoid activation, $\sigma$ as discussed prior, is defined as $\frac{1}{1 + e^{-z}}$.

The function is shaped as an S-shaped curve, in the range of $0 < \hat{y} < 1$, one of it's use cases being to compute an output as a probability of an input belonging to a given class, within binary classification.

<div align = 'center'>
<img src = '../util_images/logisticcurve.svg' width = 400>
</div>

It's derivative can be defined as:

<div align = 'center'>

$σ(z) = \frac{e^z}{e^z + 1}$

$σ(z) = (1+e^{-z})^{-1}$

$σ'(z) = - (1 +e^{-z})^{-2}(-e^{-z})$

$σ'(z) = - \frac{-e^{-z}}{(1 +e^{-z})^{2}}$

$σ'(z) = \frac{e^{-z}}{(1 +e^{-z})^{2}}$

</div>

Sigmoid can be considered a smooth function, given that it's derivative is continuous for all real values of $z$, making it proper for using optimization algorithms such as gradient descent.

It's also a monotonically increasing function, with no abrupt changes in it's sign value. This then allow for a model to better converge onto a global minima.

Regarding it's computational efficiency, given that it contains the exponential function, $e^z$, it's computationally expensive given that it requires *["hundreds of addition, subtraction, multiplication, and division instructions on a general-purpose CPU"](https://www.sciencedirect.com/topics/computer-science/sigmoid-function#:~:text=High%20computational%20complexity.,Thus%2C%20the%20computation%20is%20inefficient).*

Though, nearing $0$ and $1$, it's gradient is very small. 

This becomes a problem in deeper networks, where to get the gradients of earlier layers through backpropagation and the chain rule, we need to multiply the gradient of the output layers with earlier layers. 

Given that sigmoid yields a small gradient when it converges to near $0$ or $1$, the deeper we get into a network, the smaller a gradient will be in earlier layers, then leading to slower learning and convergence

### Softmax

> [Code](ActivationFuncs.ipynb#cell-3)


Just as was covered prior, the softmax activation function can be defined as, $\frac{e^z}{\sum e^z}$.

The range of softmax is from $0$ to $1$, outputting a probability of an input belonging to a specific class. 

It being bounded between this rasnge makes it suitable for the computation of a probability, as it also divides the exponential, $e$, raised to given class $z$, over the sum of $e$ raised to all classes $z$. 

Therefore, function requires the output of all classes to be equal to $1$, reinforcing it's use for computing a probability.

Softmax is also smooth everywhere, unlike ReLU and it's variants. This allows for backpropagation to work effectively throughout the entire network ultimately mitigating the propagation of NaN values throughout a model.

It's continuity allows for gradient descent to slowly update parameters in a continuous fashion with no abrupt changes. Being monotonically increasing then allows for easier interpretability of the learning behavior of a model.

Softmax is less computationally efficient than most activation functions such as ReLU, sigmoid, 
or tanh, but it's primarily used in the output layers of a model making this not much of an issue.

Common use cases of softmax are primarily only multi-class classification problem, being used in the final output layer. Going more complex, it can be used in attention mechanisms to compute the attention weights.

The cool thing about using $e$ in softmax, is the exponential increase in output it has in relation to an increase to inputs. 

Say we're aiming to optimize for the one hot encoded label, $[0, 1, 0, 0]$.

If a model has a normalized logit of: $[.2222, .4444, .2222, .1111]$ as the final output, it's clear that the $armgax$, despite it being greater than the rest of the values, isn't very high compared to the rest as it should be. This might make it harder for a model to get accurate predictions consistently.

But if we introduce the softmax, and we raise those values to $e$, the outputs end up being: $[0.1025,0.7573,0.1025,0.0377]$, where it's clear that the $argmax$ is closer to the true class label, which then provides an increase in accuracy as a model is trained.

### TanH

> [Code](ActivationFuncs.ipynb#cell-4)

The hyperbolic tangent resembles $\sigma$, but has a range instead between $[-1, 1]$ rather than $[0,1]$.
<br><br>
<div align = 'center'>
<img src = '../util_images/tanh.png' width = 400>
</div>
<br>
It can be defined as:

<div align = 'center'>

$TanH = \frac{e^z - e^{-z}}{e^z + {e^{-z}}}$
</div>

while it's derivative can be defined as:

<div align = 'center'>

$TanH'= \frac{(e^z + e^{-z})(e^z - e^{-z})' - (e^z - e^{-z})(e^z + e^{-z})'}{({e^z + e^{-z}})^2}$

$TanH'= \frac{(e^z + e^{-z})^2 - (e^z - e^{-z})^z}{(e^z + e^{-z})^2}$

$TanH'= \frac{(e^z + e^{-z})^2}{(e^z + e^{-z})^2} - \frac{(e^z - e^{-z})^2}{(e^z + e^{-z})^2}$

$TanH'= 1 - tanh(z)^2$

</div>

It's range is between, $-1$ and $1$ symmetric around the origin of $0,0$, which ends up yielding a steeper gradient than the $\sigma$ function.

The $TanH$ function is smooth and continuous everywhere, making it differentiable amongst it's entire range.

The smoothness enables backpropagation to effectively compute the gradients and update a model's parameters without issues of numerical instability while the continuity enables a smooth gradient descent mitigating sudden and abrupt changes in the model parameters.

Being a monotonically increasing function, there are no abrupt changes in it's sign making the learning during Backpropagation smooth and interpretable.

The computational efficiency of tanh is a burden, requiring the computation of multiple 
 `np.exp`'s, more than $\sigma$. 

### ReLU

> [Code](ActivationFuncs.ipynb#cell-5)

$ReLU$ is an activation function that's $0$ when $z < 0$ and linear when $z > 0$.

<div align = 'center'>
<img src = '../util_images/relu.png' width = 400>
</div>

Mathematically, it's defined as:

<div align = 'center'>

$ReLU(z) = \begin{cases} z, z>0 \\ 0, z<0 \end{cases}$

$ReLU(z) = max(0, z)$

</div>

while it's derivative looks like:

<div align = 'center'>

$ReLU'(z) = \begin{cases} 1, z>0 \\ 0, z<0 \end{cases}$
</div>

The range of ReLU is from $0$ to $\infty$, providing a stable output of $0$ if the input $z$ is less than $0$ and a linear output of $z$ is the input $z$ is greater than $0$. 

Given that it isn't bound to a specific range such as $\sigma$ or $tanh$, it's use case will not be for normalizing data between fixed range, which can be inefficient when calculating final outputs given a lack of interpretability. Like $\sigma$ can be used to calculate probabilities, $ReLU$ can't be interpreted as such.

But it doesn't suffer from the vanishing gradient problem. Rather it's sparse when $z < 0$ and has a consistent output when $z > 0$.

> *Though, introducing sparsity might lead to 'dying neurons', which can also inhibit learning. You typically mitigate this through a modification to ReLU called Leaky ReLU.*

$ReLU$ is not a 'smooth' function throughout it's entire range. At $0$, $ReLU$ is not differentiable, which can potentially cause problems, but it's unlikely that an input will be exactly $0$ due to the linear combination and the addition of a bias parameter, $B$ in the affine transformation prior.

In practice, the discontinuity can be easily done away with, by simply letting $ReLU$ be equivalent to $0$ when $z$ is $0$

<div align = 'center'>

```
numpy.maximum(z, 0)

'''
This returns z, when z is greater than 0. If z <= 0, the function will return 0.
'''
```
</div>

$ReLU$ is a strictly monotonically increasing function. As the inputs increase, the activations gotten with $ReLU$ will linearly increase when $z > 0$, then helping the training of a model become more predictable.

$ReLU$ is also more computationally efficient than most activation functions such as $\sigma$ or $tanh$. It's simple use of the $max()$ function and absence of Euler's $e$, makes for computationally efficient operations.

Given it's lack of suitability for output layers, with it's unbounded upper range, it's lowered computational complexity, and avoidance of the Vanishing Gradient problem, $ReLU$ is commonly used in the hidden layers of a deep network, especially when sparsity is desired.

### Leaky ReLU

> [Code](ActivationFuncs.ipynb#cell-6)

$Leaky$ $ReLU$ is an activation function based on ReLU with the difference being that when $z < 0$, it has a small gradient rather than having a $0$ gradient.
<br><br>
<div align = 'center'>
<img src = '../util_images/leakyrelu.png' width = 400>
</div>
<br>

Mathematically, it is defined as:

<div align = 'center'>

$LeakyReLU(z) = \begin{cases} z, z>0 \\ 0.01z, z<0 \end{cases}$

$LeakyReLU(z) = max(.01z, z)$

</div>

The derivative of Leaky ReLU can be defined as:

<div align = 'center'>

$LeakyReLU'(z) = \begin{cases} 1, z > 0 \\ .01, z < 0 \end{cases}$
</div>

The range of $Leaky$ $ReLU$ goes from $-\infty$ to $\infty$. Given this purely unbounded range, unlike ReLU, it isn't prone to 'dead neurons'. 

Rather having a smaller gradient of $.01$ when $z < 0$, is allows for a model to learn despite outputting negative values.

Just like $ReLU$, $Leaky$ $ReLU$ isn't smooth everywhere as it isn't differentiable when an input $z$ is equal to $0$. 

Mathematically, this discontinuity may post a problem. But inputs typically aren't $0$ given that the weighted sum involves a linear combination and the addition of a bias parameter. 

And again, can be bypassed in practice just like this:

<div align = 'center'>

```
a = np.where(z > 0, z, .01 * z)

'''
This returns z, when z is greater than 0. If z <= 0, the function will return (.01 * z).
'''
```
</div>

The function is strictly monotonically increasing with no abrupt changes in it's sign value. This then prevents the gradient descent from converging onto a local minima rather than a global one, allowing for effective learning over time.

$Leaky$ $ReLU$ is more computationally complex than $ReLU$, given the addition of a multiplication but still way more computationally efficient than $\sigma$ and $TanH$ given the absence of Euler's number, $e$.

Common use cases include activations at the hidden layers of deep networks. It's lack of computational complexity, ability to cope with vanishing gradients, lack of extreme sparsity, and unbounded range make it an improved activation function for hidden layers.

> [!IMPORTANT]
>***Feel free to checkout the files, [nn.py](nn.py) or [MiniBatchNN.py](MiniBatchNN.py) and replace the `ReLU` activation function and it's derivative `ReLU_deriv` with any of the above to test them out.***

## parameter initialization

Parameter initialization is an important factor to consider when you begin to train models, the more complex they get, the more important proper parameter initialization is.

It can mark the difference between being able to effectively learn and a model inexplicably running into NaN values or overflow / underflow errors.

With proper parameter initialization, the goal is to break symmetry in a set of weights, $W$, because if they were symmetric, they would end up learning the same set of features which would then result in the model being unable to learn complex patterns.

It's also key to initialize the parameters to values that are small enough to not cause exploding gradients, otherwise, your model would very quickly overshoot and fail to converge on an optimum set of weights.

Keep in mind, most parameter initialization techniques are typically applied to the weights, $W$, and not the bias $B$.

This is as $W$ plays a greater role in the model being able to learn complex features while $B$ is more of a shifting parameter for the activation function, that only centers the mean of a given output at a different value.

All values of $B$ can be initialized to $0$, while it's important for all weights, $W$, not to be.

### xavier initialization

Xavier initialization, propsed by Xavier Glorot and Yoshua Bengio, is a type of weight initialization that first initializes a set of weights, in this case we'll refer as matrix $W$, from a gaussian or uniform distribution.

Then, the initialized $W$, is multiplied by a factor of $\sqrt{\frac{1}{m^{l-1}}}$, where $m^{l-1}$ is the number of neurons or nodes in the previous layer.

So to initialize the weights, you'd do so as:

<div align = 'center'>

$W ～ N(0, 1)$

$W * \sqrt{\frac{1}{m^{l-1}}}$
</div>

This type of initialization, as described in the original [paper](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) is well suited, mathematically & empirically proven for the $\sigma$ activation functions.

It expects an activation function to be linear and 0 centered.

While $\sigma$ isn't exactly linear, with small inputs nearing $0$, they become linear for those respective regions making Xavier initialization well suited for the case, as it aims to reduce the variance of the weights within this region.

Of course, this then helps mitigate the issue of vanishing gradients presented by $\sigma$, though it isn't a permanent solution for all situations.

### kaiming initialization

Kaiming initialization or also known as "He initialization", is similar to Xavier initialization, with the difference that it's meant to be more well suited for the $ReLU$ activation function.

It  assumes the inputs are normalized with zero as the mean. Weights and biases are initialized from a symmetric distribution at zero. This might mean gaussian with mean 0 or uniform with mean 0.

Mathematically, it can be defined as:

<div align = 'center'>

$W ～ N(0, 1)$

$W^{(l)} = W^{(l)} \cdot \sqrt{\frac{2}{m^{(l-1)}}}$
</div>

This aims to preserve the variance of the gradients during backpropagation to avoid vanishing or exploding gradients, for the ReLU activation as He initialization initializes weights with a larger variance.

Though of course, it isn't a one-size-fits-all solution.

> [!NOTE] 
> *Read more about Kaiming Initialization [here](https://arxiv.org/pdf/1502.01852)*

You can check out an implementation of Kaiming initialization in [nn.py](nn.py) or [MiniBatchNN.py](MiniBatchNN.py), feel free to mess around and test things out.