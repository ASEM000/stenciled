
import numpy as np
import numba

def stenciled(      
                inplace = False , 
                window  = (-1,-1), 
                border  =  None, 
                parallel = False, 
                relative_indexing = True ) :

    @numba.njit
    def update_window(S,X):
        if S == (-1,-1) :return X.shape
        else : return S
    
    @numba.njit
    def get_window(X,P,S):
        # X : Array
        # P : Current point
        # S : window size
        if S == (-1,-1):
            S = X.shape; 
            XX = pad(X,border=(S[0],S[1],S[0],S[1]))
            res = ((XX[P[0]+S[0]-S[0]//2:P[0]+S[0]+(S[0]+1)//2,P[1]+S[1]-S[1]//2:P[1]+S[1]+(S[1]+1)//2]))            
            return XX[P[0]+S[0]-S[0]//2:P[0]+S[0]+(S[0]+1)//2,P[1]+S[1]-S[1]//2:P[1]+S[1]+(S[1]+1)//2]
        
        else :
            return X[P[0]-S[0]//2:P[0]+(S[0]+1)//2,P[1]-S[1]//2:P[1]+(S[1]+1)//2]
    
    
    @numba.njit
    def center(X):
        
        if relative_indexing:
            # Adjust array indices from top left to center cell
            rx,cx  = X.shape
            r0,c0 = (rx-1)//2 , (cx-1)//2
            result = np.zeros_like(X)

            left_region = X[r0:,:c0] ; result[:rx-r0,cx-c0:] = left_region
            top_region  = X[0:r0,c0:] ; result[rx-r0:,: cx-c0] = top_region
            center_region = X[r0:,c0:] ; result[:rx-r0,:cx-c0] = center_region
            top_left_region = X[:r0,:c0] ; result[rx-r0:,cx-c0:] = top_left_region

            return result
        else :
            return X
    
    @numba.njit
    def pad(X,border=(1,1,1,1),const=0):
        t,r,b,l = border
        xr,xc = X.shape
        Y = np.ones((xr+(t+b),xc+(l+r))) * const
        Y[t:xr+t,l:xc+l] = X
        return Y

    
    @numba.njit
    def apply_func(args,pt,func):

        if len(args)>1:
            #Multiple argument
            return func( center(get_window(args[0],pt,window)), *args[1:]) 
        
        elif len(args)==1 :
            #Single arguments
            return func( center(get_window(args[0],pt,window))  )
    
    
    def _stenciled(func):

        # jit the incoming function
        func=numba.njit(func) 
        
        if inplace == False :
            
            # No update in RowXColumn
            @numba.njit(parallel=parallel)
            def calculate(*args):
                
                ny,nx=args[0].shape ; 
                Y = np.zeros_like(args[0])
                
                if border == 'all' :
                    Y[:,0] = args[0][:,0] ; Y[:,-1] = args[0][:,-1] ; Y[0,:] = args[0][0,:] ; Y[-1,:] = args[0][-1,:]
                    py,px = (np.array(update_window(window,Y))+1) //2
                    padded_args = (pad(args[0] , (py,px,py,px),0),) + args[1:]
                    ny,nx=padded_args[0].shape
                    for yi in numba.prange(py+1,ny-py-1):
                        for xi in numba.prange(px+1,nx-px-1):
                            Y[(yi-py,xi-px)] = apply_func(padded_args,pt=(yi,xi),func=func)
                    return Y 

                elif border =='top':
                    Y[0,:] = args[0][0,:]
                    py,px = (np.array(update_window(window,Y))+1) //2
                    padded_args = (pad(args[0] , (py,px,py,px),0),) + args[1:]
                    ny,nx=padded_args[0].shape
                    for yi in numba.prange(py+1,ny-py):
                        for xi in numba.prange(px,nx-px):
                            Y[(yi-py,xi-px)] = apply_func(padded_args,pt=(yi,xi),func=func)
                    return Y  

                elif border =='left':
                    Y[:,0] = args[0][:,0]
                    py,px = (np.array(update_window(window,Y))+1) //2
                    padded_args = (pad(args[0] , (py,px,py,px),0),) + args[1:]
                    ny,nx=padded_args[0].shape
                    for yi in numba.prange(py,ny-py):
                        for xi in numba.prange(px+1,nx-px):
                            Y[(yi-py,xi-px)] = apply_func(padded_args,pt=(yi,xi),func=func)
                    return Y                 
                
                elif border==None:
                    py,px = (np.array(update_window(window,Y))+1) //2
                    padded_args = (pad(args[0] , (py,px,py,px),0),) + args[1:]
                    ny,nx=padded_args[0].shape
                    for yi in numba.prange(py,ny-py):
                        for xi in numba.prange(px,nx-px):
                            Y[(yi-py,xi-px)] = apply_func(padded_args,pt=(yi,xi),func=func)
                    return Y 

                
            return calculate    

        elif inplace == True:
            
            @numba.njit
            def calculate(*args):
                
                ny,nx=args[0].shape ; 
                Y = np.copy(args[0])

                if border == 'all' :
                    Y[:,0] = args[0][:,0] ; Y[:,-1] = args[0][:,-1] ; Y[0,:] = args[0][0,:] ; Y[-1,:] = args[0][-1,:]
                    py,px = (np.array(update_window(window,Y))+1) //2
                    padded_args = (pad(args[0] , (py,px,py,px),0),) + args[1:]
                    ny,nx=padded_args[0].shape
                    for yi in range(py+1,ny-py-1):
                        for xi in range(px+1,nx-px-1):
                            padded_args[0][(yi,xi)] = apply_func(padded_args,pt=(yi,xi),func=func)
                    return padded_args[0][py:-py,px:-px]

                elif border =='top':
                    Y[0,:] = args[0][0,:]
                    py,px = (np.array(update_window(window,Y))+1) //2
                    padded_args = (pad(args[0] , (py,px,py,px),0),) + args[1:]
                    ny,nx=padded_args[0].shape
                    for yi in range(py+1,ny-py):
                        for xi in range(px,nx-px):
                            padded_args[0][(yi,xi)] = apply_func(padded_args,pt=(yi,xi),func=func)
                    return padded_args[0][py:-py,px:-px]  

                elif border =='left':
                    Y[:,0] = args[0][:,0]
                    py,px = (np.array(update_window(window,Y))+1) //2
                    padded_args = (pad(args[0] , (py,px,py,px),0),) + args[1:]
                    ny,nx=padded_args[0].shape
                    for yi in range(py,ny-py):
                        for xi in numba.prange(px+1,nx-px):
                            padded_args[0][(yi,xi)] = apply_func(padded_args,pt=(yi,xi),func=func)
                    return padded_args[0][py:-py,px:-px]                 
                
                elif border== None:
                    py,px = (np.array(update_window(window,Y))+1) //2
                    padded_args = (pad(args[0] , (py,px,py,px),0),) + args[1:]
                    ny,nx=padded_args[0].shape
                    for yi in range(py,ny-py):
                        for xi in range(px,nx-px):
                            padded_args[0][(yi,xi)] = apply_func(padded_args,pt=(yi,xi),func=func)
                    return padded_args[0][py:-py,px:-px]
                
            return calculate
        
    return _stenciled
    