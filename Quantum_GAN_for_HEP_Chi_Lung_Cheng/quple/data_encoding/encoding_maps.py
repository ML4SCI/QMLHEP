from functools import reduce
import numpy as np
import sympy as sp

from quple.components.interaction_graphs import cyclic

def trial_0_qiskit(x: np.ndarray) -> float:
    if len(x) == 1:
        coeff = x[0]
    elif len(x) == 2:
        coeff = (x[0]+x[1])/np.pi
    else:
        raise ValueError('The encoding map "trial_0" supports only up to second order feature maps')
    return coeff

class trial_2_qiskit:
    def __init__(self, degree=1):
        self.degree = degree
    def __call__(self, x: np.ndarray):
        if len(x) == 1:
            coeff = x[0]**self.degree
        elif len(x) == 2:
            coeff = (x[0]+x[1])**self.degree/sp.pi
        else:
            raise ValueError('The encoding map "trial_2_qiskit" supports only up to second order feature maps')
        return coeff

class polynomial:
    def __init__(self, degree=1, scale_factor=sp.pi):
        self.degree = degree
        self.scale_factor = scale_factor
    def __call__(self, x):
        coeff = (reduce(lambda m, n: m + n, x)/len(x))**self.degree/self.scale_factor
        return coeff
    
class trial_2:
    def __init__(self, degree=1):
        self.degree = degree
    def __call__(self, x: np.ndarray):
        if len(x) == 1:
            coeff = x[0]**self.degree/sp.pi
        elif len(x) == 2:
            coeff = (x[0]+x[1])**self.degree/sp.pi**2
        else:
            raise ValueError('The encoding map "trial_2" supports only up to second order feature maps')
        return coeff

def trial_1(x: np.ndarray) -> float:
    if len(x) == 1:
        coeff = x[0]/sp.pi
    elif len(x) == 2:
        coeff = (x[0]+x[1])/sp.e**2
    elif len(x) == 3:
        coeff = (x[0]+x[1]+x[2])/sp.e**2
    elif len(x) == 4:
        coeff = (x[0]+x[1]+x[2]+x[3])/sp.e**4
    else:
        raise ValueError('The encoding map "trial_0" supports only up to third order feature maps')
    return coeff

def trial_0(x: np.ndarray) -> float:
    if len(x) == 1:
        coeff = x[0]/sp.pi
    elif len(x) == 2:
        coeff = (x[0]+x[1])/sp.pi**2
    elif len(x) == 3:
        coeff = (x[0]+x[1]+x[2])/sp.pi**2
    elif len(x) == 4:
        coeff = (x[0]+x[1]+x[2]+x[3])/sp.pi**4
    else:
        raise ValueError('The encoding map "trial_0" supports only up to third order feature maps')
    return coeff

def qiskit_default(x: np.ndarray) -> float:
    coeff = 2*x[0]/(sp.pi**2)  if len(x) == 1 else 2*reduce(lambda m, n: m * n, (np.pi-x))/(sp.pi**2)
    return coeff

def self_product(x: np.ndarray) -> float:
    """
    Function: \prod_{i=0, n} x_i
    Domain: (-1, +1)
    Range: (-1, +1)

    Args:
        x: data

    Returns:
        float: the mapped value
    """
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m * n, x)  
    return coeff

def shifted_self_product(x: np.ndarray) -> float:
    """
    Function: \prod_{i=0, n} (1-x_i)
    Domain: (-1, +1)
    Range: (0, +2)

    Args:
        x: data

    Returns:
        float: the mapped value
    """
    coeff = reduce(lambda m, n: m * n, 1-x)/2**(len(x)-1)
    return coeff

def pi_exponent(x: np.ndarray) -> float:
    """
    Function: 
    f(x) = x if n_dim = 1
    f(x) = pi^(|x_0-x_1|^2/8)/pi if n_dim = 2
    Domain: (-1, +1)
    Range: (pi**(-0.5), 1)
    
    Args:
        x: data

    Returns:
        float: the mapped value
    """
    if len(x) == 1:
        return x[0]
    elif len(x) == 2:
        return (sp.pi**(abs(x[0]-x[1])**2/8-1))
    else:
        raise ValueError('Encoding function does not allow input of dimension > 2')
        
def cosine_reciprocal(x: np.ndarray) -> float:
    """
    Function: 
    f(x) = x if n_dim = 1
    f(x) = 1/(3*cos(x_0)*cos(x_1)) if n_dim = 2
    Domain: (-1, +1)
    Range: ?
    
    Args:
        x: data

    Returns:
        float: the mapped value
    """
    if len(x) == 1:
        return x[0]
    elif len(x) == 2:
        return 1/(3*sp.cos(x[0])*sp.cos(x[1]))
    else:
        raise ValueError('Encoding function does not allow input of dimension > 2')        
    
def cosine_product(x: np.ndarray) -> float:
    """
    Function: f(x) = \prod_{i=0, n} cos(x_i)
    Domain: (-1, +1)
    Range: (-1, +1)?

    Args:
        x: data

    Returns:
        float: the mapped value
    """
    cos_x = [sp.cos(x_i) for x_i in x]
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m * n, cos_x)  
    return coeff

def modified_cosine_product(x: np.ndarray) -> float:
    """
    Function: f(x) = \prod_{i=0, n} cos(pi*(x_i+1)/2)
    Domain: (-1, +1)
    Range: (-1, +1)

    Args:
        x: data

    Returns:
        float: the mapped value
    """
    cos_x = [sp.cos(sp.pi*(x_i+1)/2) for x_i in x]
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m * n, cos_x)  
    return coeff

def modified_sine_cosine_alternate_product(x: np.ndarray) -> float:
    """
    Function: f(x) = sin(pi*x_0)*cos(pi*x_1)*sin(pi*x_2)*...
    Domain: (-1, +1)
    Range: (-1, +1)

    Args:
        x: data

    Returns:
        float: the mapped value
    """
    mix_x = [sp.cos(sp.pi*x[i]) if i%2 else sp.sin(sp.pi*x[i]) for i in range(len(x))]
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m * n, mix_x)  
    return coeff

def sine_cosine_alternate_product(x: np.ndarray) -> float:
    """
    Function: f(x) = sin(x_0)*cos(x_1)*sin(x_2)*...
    Domain: (-1, +1)
    Range: (-1, +1)?

    Args:
        x: data

    Returns:
        float: the mapped value
    """
    mix_x = [sp.cos(x[i]) if i%2 else sp.sin(sp.pi*x[i]) for i in range(len(x))]
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m * n, mix_x)  
    return coeff

def distance_measure(x: np.ndarray) -> float:
    """
    Function: f(x) = (\prod_{i<j} (x_i-x_j))/2**(n_pairs)
    Domain: (-1, +1)
    Range: (-1, +1)

    Args:
        x: data

    Returns:
        float: the mapped value
    """
    if len(x) == 1:
        coeff = x[0]
    elif len(x) == 2:
        coeff = (x[0] - x[1])/2
    else:
        pairs = cyclic(len(x))
        y = [x[a]-x[b] for a,b in pairs]
        coeff = reduce(lambda m, n: m * n, y)/2**len(pairs)  
    return coeff

def one_norm_distance(x: np.ndarray) -> float:
    """
    Calculate the one norm distance between the varaibles
    Function: f(x) = (\sum_{i<j} abs(x_i-x_j))/n_pairs
    Domain: (-1, +1)
    Range: (0, +2)

    Args:
        x: data

    Returns:
        float: the mapped value
    """
    if len(x) == 1:
        coeff = x[0]
    elif len(x) == 2:
        coeff = np.abs(x[0] - x[1])
    else:
        pairs = cyclic(len(x))
        y = [np.abs(x[a]-x[b]) for a,b in pairs]
        coeff = reduce(lambda m, n: m + n, y)/(len(pairs))
    return coeff

def two_norm_distance(x: np.ndarray) -> float:
    """
    Calculate the two norm distance between the varaibles
    Function: f(x) = (\sum_{i<j} (x_i-x_j)^2/n_pairs)**0.5
    Domain: (-1, +1)
    Range: (0, +2)

    Args:
        x: data

    Returns:
        float: the mapped value
    """
    if len(x) == 1:
        coeff = x[0]
    elif len(x) == 2:
        coeff = np.abs(x[0] - x[1])
    else:
        pairs = cyclic(len(x))
        y = [(x[a] - x[b])**2 for a,b in pairs]
        coeff = ((reduce(lambda m, n: m + n, y)/len(pairs))**(1/2))
    return coeff

def cube_sum(x: np.ndarray) -> float:
    """
    Function: f(x) = (\sum_{i=0, n} x_i^3)/n
    Domain: (-1, +1)
    Range: (-1, +1)
    
    Args:
        x: data

    Returns:
        float: the mapped value
    """
    cube_x = [x_i**3 for x_i in x]
    coeff = reduce(lambda m, n: m + n, cube_x)/len(x)
    return coeff   

def arithmetic_mean(x: np.ndarray) -> float:
    """
    Function: f(x) = (\sum_{i=0, n} x_i)/n
    Domain: (-1, +1)
    Range: (-1, +1)
    
    Args:
        x: data

    Returns:
        float: the mapped value
    """
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m + n, x)/len(x) 
    return coeff    
    
def second_moment(x: np.ndarray) -> float:
    """
    Function: f(x) = ((\sum_{i=0, n} (x_i+1)**2)/n)**(1/2)
    Domain: (-1, +1)
    Range: (0, +2)

    Args:
        x: data

    Returns:
        float: the mapped value
    """
    square_x = [(x_i+1)**2 for x_i in x]
    coeff = x[0] if len(x) == 1 else (reduce(lambda m, n: m + n, square_x)/len(x))**0.5
    return coeff   

def exponential_square_sum(x: np.ndarray) -> float:
    """
    Function: f(x) = exp((\sum_{i=0, n} x_i^2/n)-1)*2
    Domain: (-1, +1)
    Range: (2*exp(-1), +2)
    
    Args:
        x: data

    Returns:
        float: the mapped value
    """
    square_x = [x_i**2 for x_i in x]
    coeff = (sp.exp(reduce(lambda m, n: m + n, square_x)/len(x)-1)*2)
    return coeff         

def exponential_cube_sum(x: np.ndarray) -> float:
    """
    Function: f(x) = exp((\sum_{i=0, n} x_i^3/n)-1)*2
    Domain: (-1, +1)
    Range: (2*exp(-2), +2)
    
    Args:
        x: data

    Returns:
        float: the mapped value
    """
    cube_x = [x_i**3 for x_i in x]
    coeff = (sp.exp(reduce(lambda m, n: m + n, cube_x)/len(x)-1)*2)
    return coeff  

encoding_map_registry = {
    'trial_0': trial_0,
    'trial_1': trial_1,
    'poly_1': polynomial(degree=1),
    'poly_2': polynomial(degree=2),
    'poly_3': polynomial(degree=3),
    'poly_4': polynomial(degree=4),
    'qiskit_default': qiskit_default,
    'self_product': self_product,
    'modified_cosine_product': modified_cosine_product, 
    'cosine_product': cosine_product,
    'sine_cosine_alternate_product': sine_cosine_alternate_product,
    'modified_sine_cosine_alternate_product': modified_sine_cosine_alternate_product,
    'distance_measure': distance_measure,
    'one_norm_distance': one_norm_distance,
    'two_norm_distance': two_norm_distance,
    'arithmetic_mean': arithmetic_mean,
    'second_moment': second_moment,
    'cube_sum': cube_sum,
    'exponential_square_sum': exponential_square_sum,
    'exponential_cube_sum': exponential_cube_sum,
    'pi_exponent': pi_exponent,
    'cosine_reciprocal': cosine_reciprocal
}