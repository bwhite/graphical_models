import numpy as np
import itertools
import collections
import time


class FactorGraph(object):

    def __init__(self):
        self.variable_to_factors = {}  # [name] = list of factor indeces
        self.factors = []  # list of (tuple_of_names, N-d numpy array for N names)
        self.variable_states = {}  # [name] = list of variable states

    def add_factor(self, names, potential):
        potential_fn = lambda **kw: potential[tuple([kw.get(x, Ellipsis) for x in names])]
        potential_fn.potential = potential
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

    def to_libdai(self):
        out = '%d\n\n' % len(self.factors)
        self.name_to_num = {}
        self.num_to_names = {}
        cur_num = [0]
        
        def get_num(name):
            try:
                return self.name_to_num[name]
            except KeyError:
                self.name_to_num[name] = cur_num[0]
                self.num_to_name[cur_num[0]] = name
                cur_num[0] += 1
                return self.name_to_num[name]
        for names, potential in self.factors:
            out += '%s\n' % len(names)
            out += '%s\n' % (' '.join([str(get_num(x)) for x in names]))
            out += '%s\n' % (' '.join([str(len(self.variable_states[x])) for x in names]))
            out += '%d\n' % potential.potential.size
            out += ''.join(['%d %f\n' % (x, y) for x, y in enumerate(np.ravel(potential.potential, order='f'))])
            out += '\n'
        return out

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
        factor_msgs, var_msgs = collections.defaultdict(dict), collections.defaultdict(dict)
        # Parent is the first neighbor a node sends to
        factor_parent, var_parent = collections.defaultdict(lambda : None), collections.defaultdict(lambda : None)
        active_vars = set(self.variable_to_factors.keys())
        active_factors = set(range(len(self.factors)))
        inactive_vars = set()
        inactive_factors = set()

        def msg_factor(from_var, to_factor):
            print('MSG Factor[%s, %s]' % (from_var, to_factor))
            prob = 1.
            for factor_ind in set(var_msgs[from_var].keys()) - set([to_factor]):
                prob *= var_msgs[from_var][factor_ind]
            if isinstance(prob, float):
                potential = lambda **kw: prob
            else:
                potential = lambda **kw: prob[kw.get(from_var, Ellipsis)]
            factor_msgs[to_factor][from_var] = potential

        def msg_variable(from_factor, to_var):
            print('MSG Variable[%s, %s]' % (from_factor, to_var))
            variables = [self.factors[from_factor]]
            for var_name in set(factor_msgs[from_factor].keys()) - set([to_var]):
                variables.append(((var_name,), factor_msgs[from_factor][var_name]))
            unique_vars = set()
            for var_names, _ in variables:
                unique_vars.update(var_names)
            unique_vars.discard(to_var)
            unique_vars = list(unique_vars)
            out = 0.

            for config in itertools.product(*[self.variable_states[x] for x in unique_vars]):
                config = dict(zip(unique_vars, config))
                prod = 1.
                for _, v in variables:
                    prod *= v(**config)
                out += prod
            var_msgs[to_var][from_factor] = out
        while active_vars - inactive_vars or active_factors - inactive_factors:
            # Sent from variables to factors
            for var_name in active_vars - inactive_vars:
                factors = self.variable_to_factors[var_name]
                if len(factors) - len(var_msgs[var_name]) == 1 and var_name not in var_parent:
                    var_parent[var_name] = list(set(factors) - set(var_msgs[var_name]))[0]
                    msg_factor(var_name, var_parent[var_name])
                elif len(factors) == len(var_msgs[var_name]):
                    for factor in set(factors) - set([var_parent.get(var_name, None)]):
                        msg_factor(var_name, factor)
                    inactive_vars.add(var_name)
            # Sent from factors to variables
            for factor_ind in active_factors - inactive_factors:
                var_names, factor = self.factors[factor_ind]
                if len(var_names) - len(factor_msgs[factor_ind]) == 1 and factor_ind not in factor_parent:
                    factor_parent[factor_ind] = list(set(var_names) - set([x[0][0] for x in factor_msgs[factor_ind]]))[0]
                    msg_variable(factor_ind, factor_parent[factor_ind])
                elif len(var_names) == len(factor_msgs[factor_ind]):
                    for variable in set(var_names) - set([factor_parent[factor_ind]]):
                        msg_variable(factor_ind, variable)
                    inactive_factors.add(factor_ind)
        msg_probs = {}
        for var_name in var_msgs:
            prob = 1.
            for x in var_msgs[var_name].values():
                prob *= x
            msg_probs[var_name] = prob / np.sum(prob)
        return msg_probs
                

def main():
    fg = FactorGraph()
    fg.add_factor(('a', 'b'), np.array([[.3, .7], [.6, .2]]))
    fg.add_factor(('c', 'b'), np.array([[.3, .7], [.6, .2]]))
    #fg.add_factor(('a', 'b'), np.array([[.3, .7], [.6, .2]]))
    #print(fg.joint({'a': 1, 'b': 1, 'c': 0}))
    print(fg.marginalize())


def test_single():
    fg = FactorGraph()
    fg.add_factor(('a',), np.array([.7, .3]))
    out = fg.marginalize()
    np.testing.assert_equal(out['a'], np.array([.7, .3]))

    fg = FactorGraph()
    fg.add_factor(('a',), np.array([.7, .3]) / 2)
    out = fg.marginalize()
    np.testing.assert_equal(out['a'], np.array([.7, .3]))

    fg = FactorGraph()
    fg.add_factor(('a',), np.array([.7, .3]) / 2)
    fg.add_factor(('a',), np.array([.7, .3]) / 2)
    out = fg.marginalize()
    np.testing.assert_almost_equal(out['a'], np.array([0.84482759,  0.15517241]))
 


def test_symmetric():
    fg = FactorGraph()
    np.random.seed(232323)
    p = np.random.random((2, 2))
    fg.add_factor(('a', 'b'), p)
    fg.add_factor(('c', 'b'), p)
    out = fg.marginalize()
    np.testing.assert_equal(out['a'], out['c'])
    print(out)


def test_paper():
    fg = FactorGraph()
    fa = np.random.random(2)
    fb = np.random.random(2)
    fc = np.random.random((2, 2, 2))
    fd = np.random.random((2, 2))
    fe = np.random.random((2, 2))
    fg.add_factor(('x1',), fa)
    fg.add_factor(('x2',), fb)
    fg.add_factor(('x1', 'x2', 'x3'), fc)
    fg.add_factor(('x3', 'x4'), fd)
    fg.add_factor(('x3', 'x5'), fe)
    st = time.time()
    out = fg.marginalize()
    print(time.time() - st)
    print(out)
    print(fg.to_libdai())
    
    
if __name__ == '__main__':
    #test_single()
    #main()
    #test_symmetric()
    #print('Paper')
    test_paper()
