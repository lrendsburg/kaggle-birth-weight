# Theoretical quantities related to Bayesian linear regression
Derivation acording to [this blog post](https://bookdown.org/aramir21/IntroductionBayesianEconometricsGuidedTour/linear-regression-the-conjugate-normal-normalinverse-gamma-model.html#:~:text=The%20conjugate%20priors%20for%20the,align).

## Conjugate Gaussian model
- $\sigma^2\sim\text{IG}(\frac{\alpha_0}{2}, \frac{\delta_0}{2})$. Inverse Gamma prior on observation noise $\sigma^2$ with parameters $\alpha_0,\delta_0>0$
- $\beta|\sigma^2\sim\mathcal{N}(\beta_0, \sigma^2 B_0)$. Gaussian distribution on weights $\beta\in\R^d$ given observation noise with parameters $\beta_0\in\R^d$ and base covariance $B_0\in\R^{d\times d}$
- $y|X,\beta,\sigma^2\sim\mathcal{N}(X\beta, \sigma^2)$. Gaussian likelihood, linear transformation with independent Gaussian noise. Data $X\in\R^{n\times d}, y\in\R^{n}$.


## Inference (general setting)
Since the prior is chosen to be Gaussian, the posterior is of the same form as the prior with updated parameters.

**Parameters**
- $B_n= (B_0^{-1} + X^TX)^{-1}$
- $\beta_n=B_n(B_0^{-1}\beta_0 + X^Ty)$
- $\alpha_n = \alpha_0 + n$
- $\delta_n = \delta_0 + y^Ty + \beta_0^T B_0^{-1}\beta_0 - \beta_n^T B_n^{-1}\beta_n$



**Posterior distributions**
- $\sigma^2|X, y\sim\text{IG}(\frac{\alpha_n}{2}, \frac{\delta_n}{2})$. 
- $\beta|\sigma^2,X, y\sim\mathcal{N}(\beta_n, \sigma^2 B_n)$. 

**Evidence (up to a constant shift and scaling)**
$$
\log p(y) =\alpha_0\log\delta_0 - \alpha_n\log\delta_n +\log\det B_n - \log\det B_0 + 2 \log \Gamma(\frac{\alpha_n}{2}) - 2 \log \Gamma(\frac{\alpha_0}{2})
$$

**Posterior predictive**
Multivariate $t$-distribution for test points $X_0\in\R^{m \times d}, Y_0\in\R^{m}$
$$
Y_0|X_0, X, y\sim t_{\alpha_n}(X_0\beta_n, \alpha_n^{-1}\delta_n(I_m + X_0 B_n X_0^T))
$$

**Predicting confidence intervals:**
For this challenge, we need to further process the joint distribution on $Y_0$ as follows: for each coordinate, we want a confidence interval to a given coverage $\alpha$. To this end, we derive the interval based on the corresponding marginal, which is a univariate $t$-distribution with corresponding parameters. Since $t$-distributions are symmetric around their mean, we can simply define the confidence interval based on the $1/2 - \alpha/2$ and $1/2 + \alpha/2$ quantiles. 

## Inference (isotropic prior)
Under the assumption $\beta_0=0, B_0=\tau^{2}I_d$, the inference simpliefies:

**Parameters**
- $B_n= (\tau^{-2} + X^TX)^{-1}$
- $\beta_n=B_n X^Ty$
- $\alpha_n = \alpha_0 + n$
- $\delta_n = \delta_0 + y^Ty - y^TXB_nX^Ty$

**Evidence (up to a constant shift and scaling)**
$$
\log p(y) =\alpha_0\log\delta_0 - \alpha_n\log\delta_n +\log\det B_n - d\log \tau^2+ 2 \log \Gamma(\frac{\alpha_n}{2}) - 2 \log \Gamma(\frac{\alpha_0}{2})
$$

**Posterior predictive**
Multivariate $t$-distribution for test points $X_0\in\R^{m \times d}, Y_0\in\R^{m}$
$$
Y_0|X_0, X, y\sim t_{\alpha_n}(X_0\beta_n, \alpha_n^{-1}\delta_n(I_m + X_0 B_n X_0^T))
$$

## Inference (isotropic prior & SVD on $X$)
The above parameters can be computed more efficiently by using the SVD $X=U\Sigma V^T$ with $U\in\R^{n\times d}, V\in\R^{d\times d}$ orthogonal and $\Sigma=\Lambda^{1/2}\in\R^{n\times d}$, where $\Lambda=\text{diag}(\lambda_1\dotsc,\lambda_d)\in\R^{d\times d}$ is diagonal and $d\leq n$ is assumed.


**Parameters**
- $B_n = V\left(\tau^{-2} + \Lambda\right)^{-1}V^T$
- $\beta_n = V\left(\tau^{-2} + \Lambda\right)^{-1}\Lambda^{1/2}U^T y$
- $\alpha_n = \alpha_0 + n$
- $\delta_n = \delta_0 + y^Ty - y^TU \left(\tau^{-2} + \Lambda\right)^{-1}\Lambda U^Ty$

**Evidence (up to a constant shift and scaling)**
$$
\log p(y) =\alpha_0\log\delta_0 - \alpha_n\log\delta_n +\sum_{i=1}^d\log(\tau^{-2} + \lambda_i) - d\log \tau^2+ 2 \log \Gamma(\frac{\alpha_n}{2}) - 2 \log \Gamma(\frac{\alpha_0}{2})
$$

**Posterior predictive**
Multivariate $t$-distribution for test points $X_0\in\R^{m \times d}, Y_0\in\R^{m}$
$$
Y_0|X_0, X, y\sim t_{\alpha_n}(X_0\beta_n, \alpha_n^{-1}\delta_n(I_m + X_0 B_n X_0^T))
$$