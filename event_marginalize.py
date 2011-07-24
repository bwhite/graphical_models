import numpy as np
import itertools


class FactorGraph(object):

    def __init__(self):
        self.variable_to_factors = {}  # [name] = list of factor indeces
        self.factors = []  # list of (tuple_of_names, N-d numpy array for N names)
        self.variable_states = {}  # [name] = list of variable states

    def add_factor(self, names, potential):
        potential_fn = lambda **kw: potential[tuple([kw.get(x, Ellipsis) for x in names])]
        self.factors.append((tuple(names), potential_fn))
        
        for dim, name in enumerate(names):
            self.variable_to_factors.setdefault(name, []).append(len(self.factors) - 1)
            if name not in self.variable_states:
                self.variable_states[name] = range(potential.shape[dim])

    def __str__(self):
        out = []
        for node, neighbors in self.unique_edges.items():
            out.append('%s--{%s}' % (node, ';'.join(neighbors)))
        return 'graph FG {%s}' % ';'.join(out)

    def joint(self, values):
        """
        Args:
            values: Dict of the form [name] = value

        Return:
            Joint probability
        """
        p = 1.
        for names, potential in self.factors:
            p *= potential[tuple([values[name] for name in names])]
        return p

    def marginalize(self):
        msg_probs = {}
        factor_msgs, var_msgs = {}, {}
        factor_parent, var_parent = {}, {}  # Parent is the first neighbor a node sends to

        def msg_factor(from_var, to_factor):
            try:
                prob = msg_probs[from_var]
                potential = lambda **kw: prob[kw[from_var]]
            except KeyError:
                potential = lambda **kw: 1.
            factor_msgs.setdefault(to_factor, []).append(((from_var,), potential))

        def msg_variable(from_factor, to_var):
            variables = [self.factors[from_factor]] + factor_msgs.get(from_factor, [])
            unique_vars = set()
            for var_names, _ in variables:
                unique_vars.update(var_names)
            unique_vars.discard(to_var)
            unique_vars = list(unique_vars)
            out = 0.

            for config in itertools.product(*[self.variable_states[x] for x in unique_vars]):
                config = dict(zip(unique_vars, config))
                out += np.prod([v(**config) for _, v in variables], axis=0)
            try:
                msg_probs[to_var] *= out
            except KeyError:
                msg_probs[to_var] = out

        while 1:
            # Sent from variables to factors
            for var_name, factors in self.variable_to_factors.items():
                if len(factors) - len(var_msgs.set_default(var_name, {})) == 1:  # Only one factor node didn't sent
                    var_parent[var_name] = set(factors) - set(var_msgs[var_name])
                    msg_factor(var_name, var_parent[var_name])
                elif len(factors) - len(var_msgs[var_name]) == 0:  # All factors sent
                    for factor in set(factors) - set([var_parent.get(var_name, None)]):
                        msg_factor(var_name, factor)
            # Sent from factors to variables
            for factor_ind, (var_names, factor) in enumerate(self.factors):
                if len(var_names) - len(factor_msgs.set_default(factor_ind, {})) == 1:  # Only one variable node didn't sent
                    factor_parent[factor_ind] = set(var_names) - set(factor_msgs[var_name])
                    msg_variable(factor_ind, factor_parent[factor_ind])
                elif len(var_names) - len(factor_msgs.set_default(factor_ind, {})) == 0:  # All variables sent
                    for variable in set(var_names) - set([factor_parent.get(factor_ind, None)]):
                        msg_variable(factor_ind, variable)
                

def main():
    fg = FactorGraph()
    fg.add_factor(('a', 'b'), np.array([[.3, .7], [.6, .2]]))
    fg.add_factor(('c', 'b'), np.array([[.3, .7], [.6, .2]]))
    #fg.add_factor(('a', 'b'), np.array([[.3, .7], [.6, .2]]))
    #print(fg.joint({'a': 1, 'b': 1, 'c': 0}))
    fg.marginalize()
    
    
if __name__ == '__main__':
    main()

