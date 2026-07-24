import pennylane as qml
import jax
import jax.numpy as jnp
from itertools import product

def square_kernel_matrix_jax(X, kernel, assume_normalized_kernel=False):
    N = qml.math.shape(X)[0]
    if assume_normalized_kernel and N == 1:
        return qml.math.eye(1, like=qml.math.get_interface(X))

    # Compute all off-diagonal kernel values, using symmetry of the kernel matrix
    i, j = jnp.tril_indices(N)
    res = jax.vmap(kernel, in_axes=(0,0))(X[i], X[j])
    mtx = jnp.zeros((N, N))  # create an empty matrix
    mtx = mtx.at[jnp.tril_indices(N)].set(res)
    mtx = mtx + mtx.T - jnp.diag(jnp.diag(mtx))

    i, j = jnp.diag_indices_from(mtx)

    if assume_normalized_kernel:
        # Create a one-like entry that has the same interface and batching as the kernel output
        # As we excluded the case N=1 together with assume_normalized_kernel above, mtx[1] exists
        mtx = mtx.at[i,j].set(1)
    else:
        # Fill the diagonal by computing the corresponding kernel values
        mtx = mtx.at[i,j].set(kernel(X[i], X[i]))

    mtx = mtx.ravel()

    if jnp.ndim(mtx[0]) == 0:
        shape = (N, N)
    else:
        shpae = (N, N, jnp.size(matrix[0]))

    return jnp.moveaxis(jnp.reshape(jnp.stack(mtx), shape), -1, 0)

def kernel_matrix_jax(X1, X2, kernel):
    N = X1.shape[0]
    M = X2.shape[0]

    products = jnp.array(list(product(X1,X2)))
    mtx = jnp.stack(jax.vmap(kernel, in_axes=(0,0))(products[:,0,:], products[:,1,:]))

    if jnp.ndim(mtx[0]) == 0:
        return jnp.reshape(mtx, (N, M))

    return jnp.moveaxis(jnp.reshape(mtx, (N, M, qml.math.size(mtx[0]))), -1, 0)

def target_alignment_jax(
    X,
    Y,
    kernel,
    assume_normalized_kernel=False,
    rescale_class_labels=True,
):
    """Kernel-target alignment between kernel and labels."""


    K = square_kernel_matrix_jax(
        X,
            kernel,
        assume_normalized_kernel=assume_normalized_kernel,
    )

    if rescale_class_labels:
        nplus = jnp.count_nonzero(jnp.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = jnp.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = jnp.array(Y)

    T = jnp.outer(_Y, _Y)
    inner_product = jnp.sum(K * T)
    norm = jnp.sqrt(jnp.sum(K * K) * jnp.sum(T * T))
    inner_product = inner_product / norm

    return inner_product