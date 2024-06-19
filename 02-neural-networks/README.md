## into neural nets

> *[Code](nn.py)*

Note that the foundations foundations of the [logistic & softmax regression](../01-logistic-&-softmax-regression/README.md) serve as similar foundations for the architectures of neural networks, the difference being that neural networks are **deep** and **complex**.

Revisiting the model of an artificial neuron:

<div align = 'center'>
<img src = '../util_images/neuron2.png' width = 450></img>
</div>

It can be recalled that a neuron takes in a set of input features, in the image above defined as $f_i$. 

In this case, we'll be defined these inputs as $X_i$, $i$ denoting the index of the feature.

The neuron takes in the set of inputs, $X_i$, applies the affine transformation $WX + B$ and then applies an activation function, either $\frac{1}{1 + e^{-z}}$ (sigmoid) or $\frac{e^{z}}{\sum{e^z}}$ (softmax), depending if you're performing binary classification or multi-class classification, to then output a final value which we'll denote as $\hat{Y}$.

A ***neural network*** has a similar flow to it's computation, but just as mentioned earlier, it's a lot more ***deep*** and ***complex***.

What is meant by ***deep*** and ***complex*** is, it contains more ***neurons*** and more ***layers***.

Here's a diagram for reference:

<div align = 'center'>
<img align = 'center' src = '../util_images/nn.png' width = 450></img><br><br>
<em style = 'font-size:13px'>Note: Most definitely, <b>not to scale</b></em>
</div>