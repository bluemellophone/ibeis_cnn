# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
print, rrr, profile = ut.inject2(__name__, '[theano_ext]')


def get_symbol_inputs(expr_list=[]):
    if not isinstance(expr_list, list):
        expr_list = [expr_list]
    inputs_ = []
    for expr in expr_list:
        if isinstance(expr, T.Constant):
            # constants don't count as inputs
            continue
        parents = expr.get_parents()
        if len(parents) == 0:
            # no parents, this is an input
            inputs_ += [expr]
        else:
            inputs_ += get_symbol_inputs(parents)
    return inputs_


def eval_symbol(expr, inputs_to_value):
    # evaluate a expr without complaining about unused inputs
    inputs_ = get_symbol_inputs([expr])
    inputs_to_value_ = {key: inputs_to_value[key]
                        for key in inputs_ if key in inputs_to_value}
    theano_value = expr.eval(inputs_to_value_)
    return theano_value
