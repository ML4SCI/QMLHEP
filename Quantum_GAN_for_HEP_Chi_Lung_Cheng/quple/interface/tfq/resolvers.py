from typing import Union, Dict, Any
from itertools import repeat
import numbers
import numpy as np
import sympy
from sympy.core import numbers as sympy_numbers
from sympy.functions.elementary.trigonometric import TrigonometricFunction

import pandas as pd
import tensorflow as tf

from quple.utils.utils import parallel_run

_RecursionFlag = object()

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


def resolve_inputs(formulas, symbols, inputs):
    param_dicts = pd.DataFrame(inputs, columns=symbols).to_dict("records")
    return tf.convert_to_tensor(parallel_run(resolve_formulas, repeat(formulas), param_dicts))

def resolve_formulas(formulas, param_dict):
    return [resolve_formula(formula, param_dict) for formula in formulas]

def resolve_formula(formula, param_dict, recursive:bool=True, deep_eval_map=None):
    if not isinstance(param_dict, dict):
        raise ValueError("resolver must be a dictionary mapping the raw symbol"
                         " to the corresponding value")
    # Input is a pass through type, no resolution needed: return early
    value = resolve_value(formula)
    if value is not NotImplemented:
        return value
    
    # Handles 2 cases:
    # formula is a string and maps to a number in the dictionary
    # formula is a symbol and maps to a number in the dictionary
    # in both cases, return it directly.
    if formula in param_dict:
        param_value = param_dict[formula]
        value = resolve_value(param_value)
        if value is not NotImplemented:
            return value
        
    # formula is a string and is not in the dictionary.
    # treat it as a symbol instead.
    if isinstance(formula, str):
        # if the string is in the param_dict as a value, return it.
        # otherwise, try using the symbol instead.
        return resolve_formula(sympy.Symbol(formula), param_dict, recursive)
                               
    # formula is a symbol (sympy.Symbol('a')) and its string maps to a number
    # in the dictionary ({'a': 1.0}).  Return it.
    if isinstance(formula, sympy.Symbol) and formula.name in param_dict:
        param_value = param_dict[formula.name]
        value = resolve_value(param_value)
        if value is not NotImplemented:
            return value
    
    # the following resolves common sympy expressions
    if isinstance(formula, sympy.Add):
        summation = resolve_formula(formula.args[0], param_dict, recursive)
        for addend in formula.args[1:]:
            summation += resolve_formula(addend, param_dict, recursive)
        return summation
    if isinstance(formula, sympy.Mul):
        product = resolve_formula(formula.args[0], param_dict, recursive)
        for factor in formula.args[1:]:
            product *= resolve_formula(factor, param_dict, recursive)
        return product
    # for more complicated operations, need to check whether values are tf.Tensors
    is_tensor = any(isinstance(is_tensor, tf.Tensor) for is_tensor in param_dict.values())
    if isinstance(formula, sympy.Pow) and len(formula.args) == 2:
        if is_tensor:
            return tf.pow(resolve_formula(formula.args[0], param_dict, recursive),
                            resolve_formula(formula.args[1], param_dict, recursive))
        return np.power(resolve_formula(formula.args[0], param_dict, recursive),
                        resolve_formula(formula.args[1], param_dict, recursive))
    # for tf.Tensors sympy subs will not preserve the tf.Tensor, need special treatment
    # for the moment, support for trigonometric function only should suffice
    if is_tensor and isinstance(formula, TrigonometricFunction):
        ops = tf_ops_map.get(type(formula), None)
        if ops is None:
            raise ValueError("unsupported sympy operation: {}".format(type(formula)))
        return ops(resolve_formula(formula.args[0], param_dict, recursive))
                               
    if not isinstance(formula, sympy.Basic):
        # No known way to resolve this variable, return unchanged.
        return formula

    # formula is either a sympy formula or the dictionary maps to a
    # formula.  Use sympy to resolve the value.
    # note that sympy.subs() is slow, so we want to avoid this and
    # only use it for cases that require complicated resolution.
    if not recursive:
        # Resolves one step at a time. For example:
        # a.subs({a: b, b: c}) == b
        value = formula.subs(param_dict, simultaneous=True)
        if value.free_symbols:
            return value
        elif sympy.im(value):
            return complex(value)
        else:
            return float(value)
    
    if deep_eval_map is None:
        deep_eval_map = {}
    # Recursive parameter resolution. We can safely assume that value is a
    # single symbol, since combinations are handled earlier in the method.
    if formula in deep_eval_map:
        value = deep_eval_map[formula]
        if value is not _RecursionFlag:
            return value
        raise RecursionError('Evaluation of {value} indirectly contains itself.')

    # There isn't a full evaluation for 'value' yet. Until it's ready,
    # map value to None to identify loops in component evaluation.
    deep_eval_map[formula] = _RecursionFlag

    value = resolve_formula(formula, param_dict, recursive=False)
    if value == formula:
        deep_eval_map[formula] = value
    else:
        deep_eval_map[formula] = resolve_formula(value, param_dict, recursive)
    return deep_eval_map[formula]                        
    
    
def resolve_value(val: Any):
    if isinstance(val, numbers.Number) and not isinstance(val, sympy.Basic):
        return val
    elif isinstance(val, tf.Tensor):
        return val    
    elif isinstance(val, sympy_numbers.IntegerConstant):
        return val.p
    elif isinstance(val, sympy_numbers.RationalConstant):
        return val.p / val.q
    elif val == sympy.pi:
        return np.pi
    else:
        return NotImplemented