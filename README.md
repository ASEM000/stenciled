# **@stenciled**

### Simplify  writing stencil kernels powered by  Numba

#### Relative indexing
<img src="https://i.imgur.com/me5bC17.png" width="50%" />

#### Example 1 : 3x3 Mean 
```python
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


#### Example 2 : Convolution ( with parallelization support)
```python
@stenciled(window=(3,3),relative_indexing=False,parallel=True)
def convolution_3x3(X,F):return numpy.sum(X*F)
```

#### Example 3 : Linear Diffusion


<img src="https://i.imgur.com/kI69TUw.png" width="500" />

```python
@stenciled(inplace=True,window=(3,3),border='all')
def diffusion_1D(u,k): 
    return u[-1,0] + k * (u[-1,1] -2*u[-1,0] + u[-1,-1])
```
___
## Speed Comparison

#### Speed comparisons ( `@numba.stencil` vs `@stenciled` )

```python
#---------------------------------------------Test functions--------------------------------------------------------------
@numba.stencil
def numba_full(x): 
    return ( x[1,-1] + x[1,0] + x[1,1] +
             x[0,-1] + x[0,0] + x[0,1] +
            x[-1,-1] +x[-1,0] + x[-1,1]) / 9

@stenciled(window=(3,3))
def stenciled_short_hand(x) : return np.mean(x)

@stenciled(window=(3,3),parallel=True)
def stenciled_short_hand_parallel(x) : return np.mean(x)

@stenciled()
def stenciled_full(x): 
    return ( x[1,-1] + x[1,0] + x[1,1] +
             x[0,-1] + x[0,0] + x[0,1] +
            x[-1,-1] +x[-1,0] + x[-1,1]) / 9

@stenciled(parallel=True)
def stenciled_full_parallel(x): 
    return ( x[1,-1] + x[1,0] + x[1,1] +
             x[0,-1] + x[0,0] + x[0,1] +
            x[-1,-1] +x[-1,0] + x[-1,1]) / 9

@stenciled(window=(3,3))
def stenciled_full_window(x): 
    return ( x[1,-1] + x[1,0] + x[1,1] +
             x[0,-1] + x[0,0] + x[0,1] +
            x[-1,-1] +x[-1,0] + x[-1,1]) / 9

@stenciled(parallel=True,window=(3,3))
def stenciled_full_window_parallel(x): 
    return ( x[1,-1] + x[1,0] + x[1,1] +
             x[0,-1] + x[0,0] + x[0,1] +
            x[-1,-1] +x[-1,0] + x[-1,1]) / 9

```


<img src='https://i.imgur.com/xUCtA4T.png' width='70%' />

##### Observations
- Specified window parameter has major speed improvements over non specified window size (kernel inference needs speed improvements )
- `@stenciled` outperforms `@numba.stencil` for specified kernel size (window parameter)
- Parallel versions outperforms single core versions for large arrays

#### Example : Convolution in python vs parallel stenciled  vs single core stenciled

```python
#----------------------------------Parallel stenciled---------------------------------------------
@stenciled(window=(3,3),parallel=True,relative_indexing=False)
def conv2dp(X,F):return np.sum(X*F)

#----------------------------------single core stenciled---------------------------------------------
@stenciled(window=(3,3),parallel=False,relative_indexing=False)
def conv2d(X,F):return np.sum(X*F)

#----------------------------------Python --------------------------------------------
#Credits : https://github.com/Alescontrela
def convolution_2D(image, filt, bias, s=1):
    '''
    Confolves `filt` over `image` using stride `s`
        
    '''
    (n_f, n_c_f, f, _) = filt.shape # filter dimensions
    n_c, in_dim, _ = image.shape # image dimensions
    
    out_dim = int((in_dim - f)/s)+1 # calculate output dimensions
        
    # ensure that the filter dimensions match the dimensions of the input image
    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"
    
    out = np.zeros((n_f,out_dim,out_dim)) # create the matrix to hold the values of the convolution operation
    
    # convolve each filter over the image
    for curr_f in range(n_f):
        curr_y = out_y = 0
        # move filter vertically across the image
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            # move filter horizontally across the image 
            while curr_x + f <= in_dim:
                # perform the convolution operation and add the bias
                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        
    return out

```

```python
X = np.random.randint(low=0,high=1000,size=(500,500)).astype('float')*1000
F = np.array([[3,4,4],[1,0,2],[-1,0,3]],dtype='float')
B= np.array([[1]])

%timeit conv2d(X,F)
%timeit conv2dp(X,F)
%timeit convolution_2D(X.reshape(1,500,500), F.reshape(1,1,3,3), B, s=1)
_____________________________________________________________________________
77.1 ms ± 2.29 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
35.4 ms ± 364 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
3.27 s ± 38.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```
**40x - 80x** speedup

___

## Comparison with @numba.stenciled

|                                  | @numba.stencil  | @stenciled |
|:--------------------------------:|:---------------:|:----------:|
|        **Numba support**         |       ✔️        |     ✔️     |
|    **Kenrel size inference**     |       ✔️        |     ✔️     |
|      **Custom kernel size**      |       ❌        |     ✔️     |
| **Multiple arrays as arguments** |       ✔️        |     ✔️     |
|         **Update rules**         |       ❌        |     ✔️     |
|       **Parallelization[1]**       |       ✔️        |     ✔️     |
|       **Border Handeling**       | `constant` Only |     ✔️     |
|     **N-Dimensional arrays**     |       ✔️        | `2D` only  |
|  **Accepts external functions**  |       ❌        |    `✔️     |

[1] For inplace = False case only
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
