import numpy as np 

def nmf(X, k, max_iter=100, tol=1e-6):
    
    m, n = X.shape
    W = np.random.rand(m, k)
    H = np.random.rand(k, n)
    
    for _ in range(max_iter):
        W_old = W
        H_old = H
        
        # update W
        W = W * np.dot(X, H.T) / np.dot(W, np.dot(H, H.T))
        W = np.maximum(W, 0)
        
        # update H
        H = H * np.dot(W.T, X) / np.dot(W.T, np.dot(W, H))
        H = np.maximum(H, 0)
        
        # check convergence
        if np.linalg.norm(W - W_old) < tol and np.linalg.norm(H - H_old) < tol:
            break
    
    return W, H