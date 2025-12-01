import jax.numpy as jnp
import jax

def gaussian_kernel(x, y, bandwidth):
    """RBF kernel k(x, y) = exp(-||x - y||^2 / (2 * h^2))
    x: (d,)
    y: (d,)
    bandwidth: scalar
    returns: scalar
    """
    sq_dist = jnp.sum((x - y) ** 2)
    return jnp.exp(-0.5 * sq_dist / (bandwidth ** 2))

def compute_mmd2(x, y, bandwidth=1.0):
    """Compute unbiased squared MMD between two sets of samples x, y."""
    gaussian_kernel_vmap = jax.vmap(jax.vmap(gaussian_kernel, in_axes=(None, 0, None)), in_axes=(0, None, None))
    Kxx = gaussian_kernel_vmap(x, x, bandwidth)
    Kyy = gaussian_kernel_vmap(y, y, bandwidth)
    Kxy = gaussian_kernel_vmap(x, y, bandwidth)

    n = x.shape[0]
    m = y.shape[0]

    sum_Kxx = jnp.sum(Kxx) / (n * n)
    sum_Kyy = jnp.sum(Kyy) / (m * m)
    sum_Kxy = jnp.sum(Kxy) / (n * m)

    mmd2 = sum_Kxx + sum_Kyy - 2 * sum_Kxy
    return mmd2

class GradientKernel:
    def __init__(self, S_PQ, k):
        self.S_PQ = S_PQ
        self.k = jax.jit(k)
        self.dkx = jax.jit(jax.jacrev(self.k, argnums=0))
        self.dky = jax.jit(jax.jacrev(self.k, argnums=1))
        self.d2k = jax.jit(jax.jacfwd(self.dky, argnums=0))

        self.K = lambda X : jax.vmap(lambda x: jax.vmap(lambda y: self.k(x, y))(X))(X)
        self.dK1 = lambda X : jax.vmap(lambda x: jax.vmap(lambda y: self.dkx(x, y))(X))(X)
        self.d2K = lambda X : jax.vmap(lambda x: jax.vmap(lambda y: jnp.trace(self.d2k(x, y)))(X))(X)

        #for extensible sampling
        self.Kx = lambda X,x : jax.vmap(lambda y: self.k(x, y))(X)
        self.dK1x = lambda X,x : jax.vmap(lambda y: self.dkx(x, y))(X)
        self.dK2x = lambda X,x : jax.vmap(lambda y : self.dky(x, y))(X)
        self.d2Kx = lambda X,x : jax.vmap(lambda y: jnp.trace(self.d2k(x, y)))(X)

    def gram_matrix(self,X): #Gram_matrix
        K = self.K(X)
        dK = self.dK1(X)
        d2K = self.d2K(X)
        S_PQ = self.S_PQ(X)
        S_dK = jnp.einsum('ijk, ijk -> ij', dK, (S_PQ[None, :, :]))
        k_pq = d2K + S_dK + S_dK.T + K * jnp.dot(S_PQ, S_PQ.T)
        return k_pq
    



class KernelGradientDiscrepancy:
    def __init__(self, k_pq):
        self.K_pq = k_pq.gram_matrix
        self.k_pq = k_pq

    def evaluate(self, X):
        n = len(X)
        K_pq = self.K_pq(X)
        sum = 1/n * jnp.sqrt(jnp.sum(K_pq))
        return sum
    
    def square_kgd(self, X):
        n = len(X)
        K_pq = self.K_pq(X)
        sum = jnp.mean(K_pq)
        return sum
    
    def kde_KGD(self,X,num_samples=100):
        n,d = X.shape
        bandwidth = 1/jnp.sqrt(n)  
        key = jax.random.PRNGKey(0)
        key, key_idx, key_noise = jax.random.split(key, 3)
        indices = jax.random.choice(key_idx, n, shape=(num_samples,), replace=True)
        samples = jax.random.normal(key_noise, (num_samples, d)) * bandwidth + X[indices]
        samples = jax.device_put(samples, X.device)
        return self.evaluate(samples)


def k_imq(x, y, c, b, scale=1.):
    assert b > 0
    return (c**2 + (x-y).dot(x-y)/scale**2)**(-b)


def k_lin (x,y,c):
    return jnp.dot(x,y) + c**2

def a_(x,s,c):
    return (c**2 + jnp.sum((x)**2))**(s/2)

def recommended_kernel(x,y,L,alpha,beta,c):
    a_s_x = a_(x,alpha - beta,c)
    a_s_y = a_(y,alpha - beta,c)
    k_lin_xy = k_lin(x,y,c)/(k_lin(x,x,c)*k_lin(y,y,c))**0.5
    return a_s_x*(L(x,y) + k_lin_xy)*a_s_y