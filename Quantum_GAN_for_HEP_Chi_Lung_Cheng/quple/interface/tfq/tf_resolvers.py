import numbers
import numpy as np
import sympy
from sympy.core import numbers as sympy_numbers
from sympy.functions.elementary.trigonometric import TrigonometricFunction

import tensorflow as tf

tf_ops_map = {
    sympy.sin: tf.sin,
    sympy.cos: tf.cos,
    sympy.tan: tf.tan,
    sympy.asin: tf.asin,
    sympy.acos: tf.acos,
    sympy.atan: tf.atan,
    sympy.atan2: tf.atan2,
    sympy.cosh: tf.cosh,
    sympy.tanh: tf.tanh,
    sympy.sinh: tf.sinh
}

def stack(func, lambda_set, intermediate=None):
    if intermediate is None:
        return stack(func, lambda_set[1:], lambda_set[0])
    if len(lambda_set) > 0:
        new_lambda = lambda x:func(intermediate(x), lambda_set[0](x))
        return stack(func, lambda_set[1:], new_lambda)
    else:
        return intermediate
    
def resolve_value(val):
    if isinstance(val, numbers.Number) and not isinstance(val, sympy.Basic):
        return tf.constant(float(val), dtype=tf.float32)
    elif isinstance(val, (sympy_numbers.IntegerConstant, sympy_numbers.Integer)):
        return tf.constant(float(val.p), dtype=tf.float32)
    elif isinstance(val, (sympy_numbers.RationalConstant, sympy_numbers.Rational)):
        return tf.divide(tf.constant(val.p, dtype=tf.float32), tf.constant(val.q, dtype=tf.float32))
    elif val == sympy.pi:
        return tf.constant(np.pi, dtype=tf.float32)
    else:
        return NotImplemented    

def resolve_formulas(formulas, symbols):
    lambda_set = [resolve_formula(f, symbols) for f in formulas]
    stacked_ops = stack(lambda x, y:tf.concat((x, y), 0), lambda_set)
    n_formula = tf.constant([len(formulas)])
    transposed_x = lambda x: tf.transpose(x, perm=tf.roll(tf.range(tf.rank(x)), shift=1, axis=0))
    resolved_x = lambda x: stacked_ops(transposed_x(x))
    reshaped_x = lambda x: tf.reshape(resolved_x(x), 
                           tf.concat((n_formula, tf.strided_slice(tf.shape(x), begin=[0], end=[-1])), axis=0))
    transformed_x = lambda x: tf.transpose(reshaped_x(x), perm=tf.roll(tf.range(tf.rank(x)), shift=-1, axis=0))
    return transformed_x
           
def resolve_formulas_legacy(formulas, symbols):
    lambda_set = [resolve_formula(f, symbols) for f in formulas]
    stacked_ops = stack(lambda x, y:tf.concat((x, y), 0), lambda_set)
    n_formula = len(formulas)
    transformed_x = lambda x: tf.transpose(tf.reshape(stacked_ops(tf.transpose(x)), (n_formula, tf.gather(tf.shape(x), 0))))
    return transformed_x                               
    
def resolve_formula(formula, symbols):
    
    # Input is a pass through type, no resolution needed: return early
    value = resolve_value(formula)
    if value is not NotImplemented:
        return lambda x:value
    
    # Handles 2 cases:
    # formula is a string and maps to a number in the dictionary
    # formula is a symbol and maps to a number in the dictionary
    # in both cases, return it directly.
    if formula in symbols:
        index = symbols.index(formula)
        return lambda x:x[index]
                               
    # formula is a symbol (sympy.Symbol('a')) and its string maps to a number
    # in the dictionary ({'a': 1.0}).  Return it.
    if isinstance(formula, sympy.Symbol) and formula.name in symbols:
        index = symbols.index(formula.name)
        return lambda x:x[index]
    
    # the following resolves common sympy expressions
    if isinstance(formula, sympy.Add):
        addents = [resolve_formula(arg, symbols) for arg in formula.args]
        return stack(tf.add, addents)
    
    if isinstance(formula, sympy.Mul):
        factors = [resolve_formula(arg, symbols) for arg in formula.args]
        return stack(tf.multiply, factors)
    
    if isinstance(formula, sympy.Pow) and len(formula.args) == 2:
        base = resolve_formula(formula.args[0], symbols)
        exponent = resolve_formula(formula.args[1], symbols)
        return lambda x: tf.pow(base(x), exponent(x))
    
    if isinstance(formula, sympy.Pow):
        base = resolve_formula(formula.args[0], symbols)
        exponent = resolve_formula(formula.args[1], symbols)
        return lambda x: tf.pow(base(x), exponent(x))
    
    if isinstance(formula, TrigonometricFunction):
        ops = tf_ops_map.get(type(formula), None)
        if ops is None:
            raise ValueError("unsupported sympy operation: {}".format(type(formula)))
        arg = resolve_formula(formula.args[0], symbols)
        return lambda x: ops(arg(x))
    
