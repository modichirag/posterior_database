import numpy as np
import jax
from jax import custom_jvp
import jax.numpy as jnp

## Note, jax64 needs to be enabled!!
from jax import config
config.update("jax_enable_x64", True)

def jaxify_bs(bsmodel):
    '''
    Wrapper for lp function of Bridgestan model to interact with jit and grad in jax
    '''
    @custom_jvp
    def lp_item(x):
        result_shape = jax.ShapeDtypeStruct((), x.dtype)
        return jax.pure_callback(lambda s: np.array(bsmodel.lp(np.array(s))), result_shape, x)

    @lp_item.defjvp
    def lp_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
        primal_out = lp_item(x)
        tangent_out = jax.pure_callback(lambda s: np.array(bsmodel.lp_g(np.array(s))[1]), result_shape, x)
        tangent_out = jnp.matmul(tangent_out, x_dot)
        return primal_out, tangent_out
    
    lp = jax.vmap(lp_item, in_axes=[0])
    return lp, lp_item



if __name__=="__main__":

    # Example usage with jax
    from posteriordb import BSDB
    from jax import jit, grad
    
    model = BSDB(0)    
    lp, lp_item = jaxify_bs(model)

    D = model.dims
    batch = 8
    x = np.random.random(batch * D).reshape(batch, D)
    
    lp_from_bridgestan = model.lp(x)
    lp_from_jax = jit(lp)(x)
    print('Check lp')
    print(np.allclose(lp_from_bridgestan,  lp_from_jax))
          
    lp_grad_from_bridgestan = model.lp_g(x)[1]
    lp_grad_from_jax = jax.vmap(jit(grad(lp_item)), in_axes=[0])(x)
    print('Check lp grad over a batch')
    print(np.allclose(lp_grad_from_bridgestan,  lp_grad_from_jax))
    
