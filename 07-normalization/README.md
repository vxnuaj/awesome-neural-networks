## normalization

Normalization is simply the process of scaling down the inputs to a neural network by standardizing or normalizing that given set of inputs

This can be done through a variety of manners, such as min-max normalization or z-score normalization. 

By applying normalization to a set of inputs, if properly done so, you can reduce the computational resources you use, mitigate potential training instability, speed up training, and ultimately converge on the optimal set of parameters at a faster rate.

As an example, applying batch normalization, [here](AdamBatchNormNN.py), with the adam optimizer was able to yield $> 90$% accuracy in less than 150 epochs or 1500 training steps. 

The risk with not applying a type of normalization, are imbalanced weights (covariate shift), slower convergence & an increased need for more computational power, and potential trianing instability due to exploding or vanishing gradients.

### normalizing first-layer inputs

Normalizing the inputs to the first layer can be done through min-max normalization or z-score normalization, ideally taking a set of given inputs and reducing them to a range between $[-1, 1]$ or $[0, 1]$. 

For min-max normalization, this looks as:

<div align = 'center'>

$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$
</div>

which then reduces the scale of a set of inputs to between $[0, 1]$

While z-score normalization looks as:

<div align = 'center'>

$X_{\mu} = \frac{1}{m} \sum X$

$Var = \frac{1}{m} \sum (X - X_{\mu})^2$

$X_{norm} = \frac{X - X_{\mu}}{\sqrt{Var}}$
</div>

which reduces the inputs to a mean of $0$ and a $Var$ of $1$.

> [!NOTE]
> *When choosing to normalize a set of inputs through z-score normalization, think twice before settling with the ReLU activation function. Doing so, because graidents are 0 when z < 0, can introduce sparsity and dead neurons.* 
>
> *A solution could be Leaky ReLU, if you don't want this...*

Both can be viable means to normalize a set of features, but is dependent on your dataset. 

Z-score normalization is more robust to outliers in a datset, making it more suitable when a datset contains them.

### Batch Normalization

> *[Implementation of a Neural Network with Batch Normalization + Adam](AdamBatchNormNN.py)*

Batch Normalization plays a similar role to z-score normalization in the sense that they both make use of the same normalization formula to reduce a set of inputs to a mean of $0$ and a $Var$ of $1$. 

What makes Batch Normalization unique is it's ability to be used **within** the hidden layers of a neural network, to mitigate imbalanced weights and exploding / vanishing gradients.

Batch Normalization also introduces two new trainable parameters, gamma ($\gamma$) and ($\beta$), which a neural network adjusts over time to control the mean ($\mu$) and variance ($Var$) of the scaling itself.

Say the forward pass of a 2 layer neural network is defined as:

<div align = 'center'>

$Z_1 = W_1X + B_1$

$A_1 = LeakyReLU(Z_1)$

$Z_2 = W_2A_1 + B_2$

$A_2 = Softmax(Z_2)$
</div>

What Batch Normalization does, is it takes the logits, the raw outputs of a layer prior to the non-linear activation, applies z-score normalization, and then adjusts the $\mu$ and $Var$ of the normalized data, to what fits best for the model to learn the most relevant features of the input datapoints.

Let's say $BatchNorm$ is defined as:

<div align = 'center'>

$Z_{\mu} = \frac{1}{m} \sum Z$

$Var = \frac{1}{m} \sum (Z - Z_{\mu})^2$

$Z_{norm} = \frac{Z - Z_{\mu}}{\sqrt{Var}}$
</div>

where $Z$ is the input, $Z_{\mu}$ is the mean of the inputs, $Var$ is the variance of the inputs, and $Z_norm$ are the $0$ mean inputs with a unit $Var$.

This $BatchNorm$ operation would be applied right after the affine transformation $Z = WX + B$.

<div align = 'center'>

$Z_1 = W_1X + B_1$

$Z_{1norm} = BatchNorm(Z_1)$

$A_1 = LeakyReLU(Z_1)$

$Z_2 = W_2A_1 + B_2$

$Z_{2norm} = BatchNorm(Z_2)$

$A_2 = Softmax(Z_2)$ 
</div>

Now, let's talk about $\gamma$ and $\beta$.

First off, to initialize these parameters, $\gamma$ can be intialized to $1$, as that's the default $Var$ applied by the z-score normalization, and $\beta$ can be initialized to $0$ as that's the default $\mu$ applied by the z-score normalization.

By initializing those parameters to those specific values, we ensure that the model has the opportunity to maintain the default settings of the normalization. If needed, the model would have the opportunity to restore the original $Var$ and $\mu$ of the inputs if needed.

This is as $\gamma$ has the ability to scale the $Var$ of a set of inputs and $\beta$ has the ability to shift the $\mu$ of the inputs.

Now, to allow for a neural network to apply the parameters $\gamma$ and $\beta$, we apply another affine transformation, this time involving $\gamma$ and $\beta$, right after the $BatchNorm$ operation.

<div align = 'center'>

$Z_1 = W_1X + B_1$

$Z_{1norm} = BatchNorm(Z_1)$

$\tilde{Z}_{1norm} = \gamma Z_{1norm} + \beta$

$A_1 = LeakyReLU(Z_1)$

$Z_2 = W_2A_1 + B_2$

$Z_{2norm} = BatchNorm(Z_2)$

$\tilde{Z}_{2norm} = \gamma Z_{2norm} + \beta$

$A_2 = Softmax(Z_2)$
</div>

Now for one last final modification, we don't need the addition of a $B$ term in the original affine transformation any longer. This is as the $\beta$ parameter will automatically make the bias shift redundant and cancel they may cancel each other's intended effects out.

So we can remove the bias term, $B$, and the final process for Batch Normalization for a 2-layer network looks as: 

<div align = 'center'>

$Z_1 = W_1X$

$Z_{1norm} = BatchNorm(Z_1)$

$\tilde{Z}_{1norm} = \gamma Z_{1norm} + \beta$

$A_1 = LeakyReLU(Z_1)$

$Z_2 = W_2A_1$

$Z_{2norm} = BatchNorm(Z_2)$

$\tilde{Z}_{2norm} = \gamma Z_{2norm} + \beta$

$A_2 = Softmax(Z_2)$
</div>

Then, to train the model, inclusive of the parameters $\beta$ and $\gamma$, we can follow the same exact process that is done for training $W$.

That is, by taking the gradients of the loss ($L$) with respect to any of the given parameters ($\theta$) and then computing the update rule, $\theta = \theta - \alpha (\frac{∂L}{∂\theta})$.

The gradients for the parameters, now that we've included 2 batch normalization operations within each layer, are computed in a different manner than what would've been done without.

It now looks as:

<div align = 'center'>

$\frac{∂L}{\tilde{∂Z}_{2norm}} = (\frac{∂L}{∂A_2})(\frac{∂A_2}{\tilde{∂Z_{2norm}}}) = A_2 - Y_{onehot}$

$\frac{∂L}{∂\gamma_2} = (\frac{∂L}{\tilde{∂Z_{2norm}}})(\frac{∂\tilde{Z_{2norm}}}{∂\gamma_2}) = \frac{∂L}{\tilde{∂Z_{2norm}}} * Z_{2norm}$

$\frac{∂L}{∂\beta_2} = (\frac{∂L}{\tilde{∂Z_{2norm}}})(\frac{\tilde{∂Z_{2norm}}}{∂\beta_2}) = \sum \frac{∂L}{\tilde{Z_{2norm}}}, axis = 1, keepdims = True$

$\frac{∂L}{∂Z_2} = (\frac{∂L}{\tilde{∂Z_{2norm}}}) (\frac{\tilde{∂Z_{2norm}}}{∂Z_{2norm}})(\frac{∂Z_{2norm}}{∂Z_2}) = \frac{∂L}{\tilde{∂Z_{2norm}}} * \gamma_2 * \frac{1}{|{\sigma_2}|}$

$\frac{∂L}{∂W_2} = (\frac{∂L}{∂Z_2})(\frac{∂Z_2}{∂W_2}) = (\frac{∂L}{∂Z_2}) \cdot A_1^T$

$\frac{∂L}{\tilde{∂Z_{1norm}}} = (\frac{∂L}{∂Z_2})(\frac{∂Z_2}{A_1})(\frac{∂A_1}{∂Z_{1norm}}) = W_2^T \cdot \frac{∂L}{∂Z_2} * ReLU_{deriv}(\tilde{Z_{1norm}})$

$\frac{∂L}{\gamma_1} = (\frac{∂L}{\tilde{∂Z_{1norm}}})(\frac{\tilde{∂Z_{1norm}}}{∂\gamma_1}) = (\frac{∂L}{\tilde{Z_{1norm}}}) * Z_{1norm}$

$\frac{∂L}{∂\beta_1} = (\frac{∂L}{\tilde{∂Z_{1norm}}})(\frac{\tilde{∂Z_{1norm}}}{∂\beta_1}) = \sum{\frac{∂L}{∂\tilde{Z}_{1norm}}}, axis = 1, keepdims = True$

$\frac{∂L}{∂Z_1} = (\frac{∂L}{\tilde{∂Z_{1norm}}}) (\frac{\tilde{∂Z_{1norm}}}{∂Z_{1norm}})(\frac{∂Z_{1norm}}{∂Z_1}) = \frac{∂L}{\tilde{∂Z_{1norm}}} * \gamma_1 * \frac{1}{|{\sigma_1}|}$

$\frac{∂L}{∂W_1} = (\frac{∂L}{∂Z_{1}})(\frac{∂Z_1}{∂W_1}) = (\frac{∂L}{∂Z_1}) \cdot X^T$<br><br>
<em style = 'font-size:12px'> Simpler derivations are rightmost... this took forever</em>
</div>

And as mentioned before, to update the $\gamma$ and $\beta$, a simple update rule can be applied, of $\theta = \theta - \alpha (\frac{∂L}{∂\theta}$

> [!NOTE]
> *If you're using a different optimizer, such as Adam or RMSprop, you'd apply them in an equivalent manner here, as was done without BatchNorm, taking the EWA of gamma and beta, to then compute a weight decay and / or momentum.*

The key rationale behind applying Batch normalization, as mentioend in the original paper, is to mitigate what's called ***covariate shift***.

Covariate shift, refers to the dynamically changing scale ($\mu$ and $Var$) of inputs to each layer, to the point where the weight updates for each layer, continuously have to shift their magnitudes to make up for these changing scales. 

This constant changing of scales, which can widly vary amongst layers, can then slow down training as the parameters of the model must adjust to keep up with teh $Var$ and $\mu$ of each of a set of inputs.

Normalizing the inputs to each layer, mitigates this covariate shift and allows for a model to train increasingly faster.

Batch Normalization can also introduce a regularization effect, when introduced alongside mini-batch rather than full-batch gradient descent.

Given that the $\mu$ and $Var$ are computed over over a set of samples in a current batch within the forward pass, computing them for each minibatch will introduce a slight degree of stochasticity, forcing the weights to continuously adapt and make up for the frequently changing $\mu$ and $Var$.

> [!NOTE]
> *Check out an implementation of Batch Normalization, with the Adam Optimizer, [here](AdamBatchNormNN.py)*
