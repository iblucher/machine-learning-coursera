# Machine Learning by Stanford University

## Week 1

### Introduction

- Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed. A more formal definition is that Machine Learning is a well-posed learning problem: a computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.
- Supervised Learning is the ML task of inferring a function from **labeled** training data, which consists of training examples (pairs of inputs and corresponding desired output values, which means we are given the "right answers" to learn from). Examples of supervised learning algorithms are regression (predict continuous valued output) and classification (predict discrete valued output).
- Unsupervised learning is the ML task of inferring a function to describe hidden structure and information from data that is **not labeled**. Examples of unsupervised learning algorithms are clustering (find hidden patterns and grouping in data) and non-clustering or "the algorithm that solves the cocktail party problem" (find structure in a chaotic environment such as identifying individual voices and music from a mesh of sounds at a cocktail party).

### Linear Regression with One Variable

- We're now going to establish notations for the aforementioned training examples of a training set of a supervised learning algorithm. We'll use $x^{(i)}$ to denote the input variables (features) and $y^{(i)}$ to denote the output variables (targets) we are trying to predict. The pair $(x^{(i)}, y^{(i)})$ is a training example. A dataset is a list of $m$ training examples.
- Our goal is, given a training set, to learn a function $h$ (hypothesis function) that maps X to Y, so that $h(x)$ is a good predictor for the corresponding value of $y$.  
- We can measure the accuracy of out hypothesis function by using a **cost function** . Our cost function is a squared error function and looks like this: $J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m}(h_ \theta (x^{(i)}) - y^{(i)})^2$. Keeping in mind that our hypothesis function is $h_ \theta (x^{(i)}) = \theta_0 + \theta_1x^{(i)}$, we want to choose $\theta_0, \theta_1$ that minimizes our cost function $J(\theta_0, \theta_1)$.
- With the linear regression algorithm, our objective is to fit a straight line (defined by $h_\theta (x)$) through our training data set, which is scattered in the x-y plane. The best possible line we can get will be the one with minimal cost function.
- To find the minimal cost function, we have to find the parameters that minimize it. We do that with the **gradient descent** algorithm. Essentially, we repeat $\theta_ j := \theta_ j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)$ until convergence. The derivative gives us a direction to move towards and the learning rate $\alpha$ gives us the size of each step along the slope. We simultaneously update the parameters after each iteration.
- "Batch" Gradient Descent: each step of the algorithm uses all of the training examples.
- When specifically applied to the case of linear regression, we can use new forms of the gradient descent equations for each parameter:
$\theta_ 0 := \theta_ 0 - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})$ $\theta_ 1 := \theta_ 1 - \alpha \frac{1}{m} \sum_{i=1}^{m}((h_\theta(x^{(i)}) - y^{(i)})x^{(i)})$


## Week 2

### Multivariate Linear Regression

- Since we can improve on the number of features our algorithm uses, we need to introduce new notations. We'll use $x_j^{(i)}$ for the value of the $j^{th}$ feature in the $i^{th}$ training example. $x^{(i)}$ is now an n-dimensional vector with the features of the $i^{th}$ training example. $n$ is the number of features. $m$ is the number of training examples.
- The new form of the hypothesis function is $h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_n x_n$. We can represent our function through matrix multiplication as $h_\theta(x) = \theta^Tx$ where $\theta$ and $x$ are n+1-dimensional vectors.
- The gradient descent algorithm haas the same form, but we repeat the math for our n features:
$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$ for j := 0...n and n >= 1.
- To speed up gradient descent we can try having each  of our input variables in the same range of values. Roughly $-1 \leq x^{(i)} \leq 1$. For that we have two techniques: **feature scaling** involves dividing the input values by the range of the input variable, resulting in a new range of just 1; and **mean normalization** which involves subtracting the average value for an input variable and dividing by the standard deviation, resulting in a new average value for the input of just 0.
- Also important for the gradient descent algorithm is the value of the learning rate $\alpha$. To see if the gradient descent algorithm is working correctly, plot the value of the cost function $J(\theta)$ over the number of iterations of gradient descent. If the value of $J(\theta)$ ever increases, the value of $\alpha$ should probably be decreased. It has been proven mathematically that for a sufficiently small $\alpha$, $J(\theta)$ should decrease on every iteration. To summarize, if $\alpha$ is too small, the convergence of $J(\theta)$ can be very slow and if $\alpha$ is too large, $J(\theta)$ may not decrease on every iteration and also may not converge.
- We can change the behavior or curve of our hypothesis function so that it fits more complexly distributed data with a **polynomial regression**, which means that our hypothesis function could be quadratic, cubic or a square root function instead of just linear. We just need to keep in mind that when choosing features this way, feature scaling becomes very important.

### Computing Parameters Analytically

- Beyond the gradient descent algorithm there is a second way of minimizing the cost function, **the "Normal Equation" method**. In this method we minimize $J$ by explicitly taking its derivatives with respect to the $\theta_j$'s and setting them to zero. The formula for this method is: $\theta = (X^TX)^{-1}X^T y$. There is no need to do feature scaling with this method.
- In comparison to the gradient descent algorithm, in the normal equation method there is no need to choose a learning rate $\alpha$ and no need to iterate, but it could become very slow if the number of features $n$ is very large as calculating the inverse of a large matrix is very costly to compute ($O(n³)$ complexity). If n exceeds 10⁴, it might be a good idea to go from a normal solution to an iterative process.
