import numpy as np


class FactorGraph(object):

    def __init__(self):
        self.variable_to_factors = {}  # [name] = list of factor indeces
        self.factors = []  # list of (tuple_of_names, N-d numpy array for N names)

    def add_factor(self, names, potential):
        self.factors.append((tuple(names), potential))
        for name in names:
            self.variable_to_factors.setdefault(name, []).append(len(self.factors) - 1)

    def __str__(self):
        out = []
        for node, neighbors in self.unique_edges.items():
            out.append('%s--{%s}' % (node, ';'.join(neighbors)))
        return 'graph FG {%s}' % ';'.join(out)

    def marginalize(self, name):

        def inner_var(cur_name, parent=None):
            out = []
            for factor_ind in self.variable_to_factors[cur_name]:
                if factor_ind != parent:
                    out.append(inner_factor(cur_ind=factor_ind, parent=cur_name))
            return np.prod(out, axis=0)

        def inner_factor(cur_ind, parent=None):
            out = []
            names, potential = self.factors[cur_ind]
            for node_name in names:
                if node_name != parent:
                    out.append(inner_var(cur_name=node_name, parent=cur_ind))
            return sum_var(names.index(parent), potential) * np.prod(out)
        inner_var(name)

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


def sum_var(ind, potential):
    potential = potential.copy()
    for x in range(potential.ndim - 1, ind, -1):
        potential = potential.sum(axis=x)
    for x in range(ind):
        potential = potential.sum(axis=0)
    return potential


def main():
    fg = FactorGraph()
    fg.add_factor(('a', 'b'), np.array([[.3, .7], [.6, .2]]))
    fg.add_factor(('c', 'b'), np.array([[.3, .7], [.6, .2]]))
    #fg.add_factor(('a', 'b'), np.array([[.3, .7], [.6, .2]]))
    #print(fg.joint({'a': 1, 'b': 1, 'c': 0}))
    
    
if __name__ == '__main__':
    main()
