# **@stenciled**

### Simplify  writing stencil kernels powered by  Numba


#### Example 1 : 3x3 Mean 
```python=1
# ---------------------------------- style 1 ----------------------------------
@stenciled()
def mean(x):
    return ( x[1,-1] + x[1,0] + x[1,1] +
             x[0,-1] + x[0,0] + x[0,1] +
             x[-1,-1] +x[-1,0] + x[-1,1]) / 9

# ---------------------------------- style 2 ----------------------------------
@stenciled(window=(3,3))
def mean(x): 
    return numpy.mean(x)   
```


#### Example 2 : Convolution 
```python=1
@stenciled(window=(3,3))
def convolution_3x3(X,F):return numpy.sum(X*F)
```

#### Example 2 : Linear Diffusion



$\large \frac{\partial u}{\partial t}= \nu \frac{\partial^2 u}{\partial x^2} \Longrightarrow u_{i}^{n}=u_{i}^{n-1}+\frac{\nu\Delta t}{\Delta x^2}(u_{i+1}^{n-1}-2u_{i}^{n-1}+u_{i-1}^{n-1})$ 


```python=1
@stenciled(inplace=True,window=(3,3),border='all')
def diffusion_1D(u,k): 
    return u[-1,0] + k * (u[-1,1] -2*u[-1,0] + u[-1,-1])
```


___

## Comparison with @numba.stenciled

|                                  | @numba.stencil  | @stenciled |
|:--------------------------------:|:---------------:|:----------:|
|        **Numba support**         |       ✔️        |     ✔️     |
|    **Kenrel size inference**     |       ✔️        |     ✔️     |
|      **Custom kernel size**      |       ❌        |     ✔️     |
| **Multiple arrays as arguments** |       ✔️        |     ✔️     |
|         **Update rules**         |       ❌        |     ✔️     |
|       **Parallelization***       |       ✔️        |     ✔️     |
|       **Border Handeling**       | `constant` Only |     ✔️     |
|     **N-Dimensional arrays**     |       ✔️        | `2D` only  |
|  **Accepts external functions**  |       ❌        |    `✔️     |


___

## Notebook examples

```
1) Basic operations
    - Identity
    - Mean filter
    - Laplacian filter
2) Border options
    - Keep all borders
    - Keep left border
    - Keep Top border
    - No borders  (default)
3) Update options (inplace  updates)
    - Linear convection
    - Non linear convection
    - 1D Diffusion
4) Window options
5) Using Functions 
    - Convolution
    - Maxpool 
```

**Credits : Mahmoud Asem - February 2021**
