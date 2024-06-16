import os
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp

print("Using jax", jax.__version__)

# defining a basic jnp array
a = jnp.zeros((2, 5), dtype=jnp.float32)
print(a)
# use a range here, same as numpy
b = jnp.arange(6)
print(b)
# print the class - should be a DeviceArray
# note that the call is b.devices() not be.device() - think it is a jax version change between 0.3.x and 0.4.x
print(b.__class__, b.devices())

# moving the arrays to another device
b_cpu = jax.device_get(b)
print(b_cpu.__class__)

b_gpu = jax.device_put(b_cpu)
print(f'Device put: {b_gpu.__class__} on {b_gpu.devices()}')

# ok so this doesnt work on local laptop, but JAX actually handles moving
# devices for you
print(b_cpu + b_gpu)

jax.devices()

# a major note - JAX does asynchronous dispatch, similar to torch
# so, it returns a placeholder array, and fills values in later

# Device Arrays are immutable!! so we use the .at[idx].set[val] to achieve the 
# roughly same function
# and like yes this is inefficient, but JIT helps with this, to be seen
b_new = b.at[0].set(1)
print('Original array:', b)
print('Changed array:', b_new)

# ah the whole random seed thing
# so this derives from JAX requiring no side effects from functions
# so if we need random elements, you explicitly pass it in
rng = jax.random.PRNGKey(42)
jax_random_number_1 = jax.random.normal(rng)
jax_random_number_2 = jax.random.normal(rng)
print('JAX - Random number 1:', jax_random_number_1)
print('JAX - Random number 2:', jax_random_number_2)

# Typical random numbers in NumPy
np.random.seed(42)
np_random_number_1 = np.random.normal()
np_random_number_2 = np.random.normal()
print('NumPy - Random number 1:', np_random_number_1)
print('NumPy - Random number 2:', np_random_number_2)

# so ya, the above is not great, and we can use random.split here to make subkeys

rng, subkey1, subkey2 = jax.random.split(rng, num=3)  # We create 3 new keys
jax_random_number_1 = jax.random.normal(subkey1)
jax_random_number_2 = jax.random.normal(subkey2)
print('JAX new - Random number 1:', jax_random_number_1)
print('JAX new - Random number 2:', jax_random_number_2)


# jax meant to be highly functional - don't do side effects, just input output, no global variables etc.
# jax traces to jaxpr intermediate language. this requires the known input/output shapes, example:

def simple_graph(x):
    x = x + 2
    x = x ** 2
    x = x + 3
    y = x.mean()
    return y

inp = jnp.arange(3, dtype=jnp.float32)
print("inp:", inp)
print("out:", simple_graph(inp))

print(jax.make_jaxpr(simple_graph)(inp))

global_list = []

# Invalid function with side-effect
def norm(x):
    global_list.append(x)
    x = x ** 2
    n = x.sum()
    n = jnp.sqrt(n)
    return n

print(jax.make_jaxpr(norm)(inp))


# Automatic Differentiation
# jax.grad! 
# you do not need to do things like loss.backward() to compute gradients
# bc jax works direct with functions. so, jax.grad takes in a function as it's
# arg. 

grad_function = jax.grad(simple_graph)
gradients = grad_function(inp)
print('Gradient', gradients)

print('gradient jaxpr', jax.make_jaxpr(grad_function)(inp))

# pretty sweet that you can just see the compute graph
# great, but what if we want output and grad? good news, JAX thought of that
val_grad_function = jax.value_and_grad(simple_graph)
print(val_grad_function(inp))

# cool cool, but then what if we want to track all the grads through a net?
# pytrees!

# now, to play some dark souls with jit
# two ways, direct wrap func lor with decroator
jitted_function = jax.jit(simple_graph)
# or like:
# @jax.jit
# def simple_graph(x):

# Create a new random subkey for generating new random values
rng, normal_rng = jax.random.split(rng)
large_input = jax.random.normal(normal_rng, (1000,))
# Run the jitted function once to start compilation
_ = jitted_function(large_input)

# there is more in tutorial to 

# ok time t
