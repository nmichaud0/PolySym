from polysym.evaluation import rmse, mse
from polysym.torch_operators_2 import Operators
from polysym.utils import (seed_everything,
                           get_logger,
                           _RandConst,
                           compile_tree,
                           _hill_climb_constants,
                           _extract_ephemerals,
                           _evaluate_worker,
                           scale_data,
                           _round_floats)
from polysym.eval_rank import is_valid_tree
import sympy as sp
import torch
from torch import Tensor
from deap import gp, base, creator, tools
from typing import Callable, Union, List, Literal
import logging
import warnings
from multiprocessing import Pool, cpu_count
from itertools import zip_longest
import copy
import operator
import random
import multiprocessing as _mp
import numpy as np
from IPython.display import display, Math

try:
    # On Linux switch to `spawn` exactly once.
    if _mp.get_start_method(allow_none=True) != 'spawn':
        _mp.set_start_method('spawn', force=True)
except RuntimeError:
    # start‑method was already set by the host programme – ignore
    pass

class Configurator:
    def __init__(self, X3d: Tensor,
                       X2d: Tensor,
                       y: Tensor,
                       max_complexity: int = 5,
                       min_complexity: int = 1,
                       max_iter: int = 1000,
                       rerun_iter: int = 10,
                       scale: bool = False,
                       stopping_criterion: Union[float, bool]=False,
                       labels_3d: list[str] = None,
                       labels_2d: list[str] = None,
                       min_max_constants: tuple[int, int] = (-100, 100),
                       opt_sigma: float = .2,
                       opt_steps: int = 100,
                       optimize_ephemerals: bool = True,
                       operators: Operators = None,
                       fitness_fn: Callable = rmse,
                       fitness_obj: Literal[-1, 1] = -1,
                       pop_size: int = 100,
                       tournament_size: int = 3,
                       cxpb: float = .5,
                       mutpb: float = .2,
                       ngsa2_alpha: float=.01,
                       seed: Union[int, None] = None,
                       verbose: int = 0,
                       workers: int = -1
                   ):

        self.X3d = X3d  # shall be of shape (n_obs, variables, time)
        self.X2d = X2d  # shape (n_obs, variables)
        self.y = y
        self.norm_stats = None

        if scale:
            self.X3d, self.X2d, self.y, self.norm_stats = scale_data(X3d, X2d, y)

        self.max_depth = max_complexity
        self.min_depth = min_complexity

        assert self.max_depth >= self.min_depth, f'Max complexity must be heigher or equal to min complexity, got: {self.max_depth}'

        self.max_iter = max_iter
        self.stopping_criterion = stopping_criterion
        if (not isinstance(self.stopping_criterion, float)) and (not (self.stopping_criterion is False)):
            raise ValueError(f"stoping criterion must be either float or False, got: {stopping_criterion}")

        self.rerun_iter = rerun_iter

        self.scale = scale

        assert type(labels_2d) == type(labels_3d), f'If setting labels, must set both for 3d and 2d inputs, got types {type(labels_2d)=}, {type(labels_3d)=}'

        self.labels_3d = labels_3d
        self.labels_2d = labels_2d
        self.labels = labels_2d + labels_3d if labels_2d is not None else None

        if labels_2d is not None:
            assert self.X3d.shape[1] == len(self.labels_3d)
            assert self.X2d.shape[1] == len(self.labels_2d)

        self.min_constant = min_max_constants[0]
        self.max_constant = min_max_constants[1]
        self.opt_sigma = opt_sigma * (self.max_constant - self.min_constant)
        self.opt_steps = opt_steps

        self.optimize_ephemerals = optimize_ephemerals

        self.fitness_fn = fitness_fn
        self.fitness_obj = fitness_obj
        self.worst_fitness = float('-inf') if fitness_obj == 1 else float('inf')
        self.pop_size = pop_size
        self.hof_size = pop_size // 4  # 25% hof size
        self.tournament_size = tournament_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngsa2_alpha = ngsa2_alpha
        self.seed = seed
        self.verbose = verbose
        self.workers = workers
        self.cpu_count = cpu_count()

        if workers >= self.cpu_count or workers == -1:
            self.workers = self.cpu_count

        if self.seed is not None:
            seed_everything(seed)

        if operators is None:
            self.operators = Operators(select_all=True)
        else:
            self.operators = operators

        # If y has 3dim then get vector output else scalar
        self.objective = y.dim()

        self.n_obs = X3d.shape[0]

        # Check dimensionality
        if not (X3d.shape[0] == X2d.shape[0] == y.shape[0]):
            raise ValueError(
            f"All input tensors must have the same number of samples (observations): "
            f"X3d.shape[0]={X3d.shape[0]}, X2d.shape[0]={X2d.shape[0]}, y.shape[0]={y.shape[0]}"
            )
        
        assert self.objective <= 2, f"Only single-output regression is supported (y.dim()={self.objective} > 2)."

        # Prepare both batched and unbatched inputs
        self.inputs = []
        self.hill_inputs = []
        idx_20p = self.n_obs // 5  # idx 20 percent
        T = self.X3d.shape[2]
        for i in range(self.X2d.shape[1]):
            self.inputs.append(self.X2d[:, i].share_memory_())
            self.hill_inputs.append(self.X2d[:idx_20p, i].share_memory_())
        """for i in range(self.X2d.shape[1]):
            x = self.X2d[:, i].unsqueeze(1).expand(-1, T)
            self.inputs.append(x.share_memory_())
            self.hill_inputs.append(x[:idx_20p].share_memory_())"""

        for i in range(self.X3d.shape[1]):
            self.inputs.append(self.X3d[:, i, :].share_memory_())
            self.hill_inputs.append(self.X3d[:idx_20p, i, :].share_memory_())

        self.y_hill = self.y[:idx_20p]

        self.pset = self._build_primitives()
        self.toolbox = self._setup_gp()
        self.hof = tools.HallOfFame(maxsize=self.hof_size)

        self.subs = {}
        if self.labels is not None:
            for var_sym, label in zip(self.symbols, self.labels):
                self.subs[var_sym] = sp.symbols(label)

        self.best_per_depth = {}
        self.best_per_depth = {depth+1: (self.worst_fitness, None, None) for depth in range(self.max_depth)}
        self.all_depths = list(range(self.max_depth))
        self.best_expr = None
        self.best_compiled = None
        self.best_fit = None
        self.best_depth = None
        self.fitted = False

        self.depth_weights = {d: 2 ** (d-self.min_depth) for d in range(1, self.max_depth+1)}
        self.depth_quota = self._compute_depth_quota(self.pop_size)

        if self.verbose == 0:
            level = logging.NOTSET
        elif self.verbose == 1:
            level = logging.INFO
        elif self.verbose == 2:
            level = logging.DEBUG
            warnings.warn(f'It is recommended to use single-threaded logic with DEBUG verbose level')
        else:
            warnings.warn(f'Verbose level not set to [1, 2, 3] – got {self.verbose=}\nSetting verbose to 0')
            level = logging.NOTSET

        self.logger = get_logger(level=level)

        self.threshold = 1e6 * -self.fitness_obj
        self.percentile = 30 if self.fitness_obj == -1 else 70

    def validate(self, ind):
        return is_valid_tree(ind, self.objective)

    def compare_func(self, a, b):

        if self.fitness_obj == -1:
            return a < b
        else:
            return a > b

    def best_func(self, a):

        if self.fitness_obj == -1:
            return min(a)
        else:
            return max(a)

    def _compute_depth_quota(self, n: int) -> dict[int, int]:
        """Desired number of individuals per depth for a pop of size n."""
        w = np.array(list(self.depth_weights.values()), dtype=float)
        w /= w.sum()
        counts = (n * w).round().astype(int)
        while counts.sum() < n:
            counts[np.argmax(w)] += 1
        while counts.sum() > n:
            counts[np.argmin(w)] -= 1
        return dict(zip(self.depth_weights, counts))

    def _make_individuals(self, depth: int, n: int=1):

        pop_fn = getattr(self.toolbox, f'population_depth_{depth}')
        return pop_fn(n=n)

    def _balanced_population(self, n: int):
        pop = []
        for depth, k in self.depth_quota.items():
            pop.extend(self._make_individuals(depth, k))
        random.shuffle(pop)
        return pop

    def _rebalance(self, pop: list):
        counts = {d: 0 for d in self.depth_quota}
        keep = []
        for ind in pop:
            d = ind.height
            if d in self.depth_quota and counts[d] < self.depth_quota[d]:
                keep.append(ind)
                counts[d] += 1

        for d, need in self.depth_quota.items():
            deficit = need - counts[d]
            if deficit > 0:
                keep.extend(self._make_individuals(d, deficit))

        random.shuffle(keep)
        return keep[:self.pop_size]

    def _setup_gp(self):

        if "Fitness" not in creator.__dict__:
            creator.create("Fitness", base.Fitness, weights=(float(self.fitness_obj),))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness, dim_mismatch=None)

        tb = base.Toolbox()
        tb.register('expr_init', gp.genHalfAndHalf, pset=self.pset, min_=self.min_depth, max_=self.max_depth)
        tb.register("individual", tools.initIterate, creator.Individual, tb.expr_init)
        tb.register("population", tools.initRepeat, list, tb.individual)

        def _wrap_population(generate):
            def wrapper(*args, **kwargs):
                pop_ = []
                i = 0
                n = kwargs['n']
                while len(pop_) < n:
                    i += 1
                    batch = generate(n=n - len(pop_), *args)
                    for ind in batch:
                        if self.validate(ind):
                            pop_.append(ind)
                            if len(pop_) >= n:
                                break
                return pop_

            return wrapper

        for d in range(1, self.max_depth+1):
            tb.register(
                f'expr_depth_{d}',
                gp.genHalfAndHalf,
                pset = self.pset,
                min_=d, max_=d
            )

            tb.register(
                f'individual_depth_{d}',
                tools.initIterate,
                creator.Individual,
                getattr(tb, f'expr_depth_{d}')
            )

            tb.register(
                f'population_depth_{d}',
                tools.initRepeat,
                list,
                getattr(tb, f'individual_depth_{d}')
            )

            tb.decorate(f'population_depth_{d}', _wrap_population)

        tb.register('clone', copy.deepcopy)

        tb.register('mutate', gp.mutUniform, expr=tb.expr_init, pset=self.pset)

        # tb.register("evaluate", _evaluate)
        tb.register('evaluate', lambda ind: _evaluate_worker(ind,
                                                             self.inputs,
                                                             self.hill_inputs,
                                                             self.pset,
                                                             self.fitness_fn,
                                                             self.y,
                                                             self.y_hill,
                                                             self.worst_fitness,
                                                             self.objective,
                                                             self.optimize_ephemerals,
                                                             self.fitness_obj,
                                                             self.threshold,
                                                             self.opt_steps,
                                                             self.opt_sigma,
                                                             self.ngsa2_alpha))

        tb.register("select", tools.selTournament, tournsize=self.tournament_size)
        tb.register("mate", gp.cxOnePoint)

        tb.decorate('population', _wrap_population)
        tb.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=self.max_depth))
        tb.decorate('mutate', gp.staticLimit(key=operator.attrgetter('height'), max_value=self.max_depth))

        return tb

    def _build_primitives(self):
        """Create symbols and DEAP PrimitiveSet with proper variable and constant terminals."""
        # Number of 2D and 3D variables
        n2, n3 = self.X2d.shape[1], self.X3d.shape[1]
        arity = n2 + n3

        # Initialize the PrimitiveSet with correct arity
        pset = gp.PrimitiveSet("MAIN", arity)

        # Add your variable terminals (x0... and v0...)
        sy2 = [sp.symbols(f"x{i}") for i in range(n2)]
        sy3 = [sp.symbols(f"v{j}") for j in range(n3)]
        self.symbols = sy2 + sy3
        for sym in self.symbols:
            pset.addTerminal(sym, name=str(sym))

        # Add ephemeral constant generator
        randc = _RandConst(self.min_constant, self.max_constant)
        pset.addEphemeralConstant("randc", randc)

        # Add operator primitives
        for name, fn in self.operators.unary_operators.items():
            pset.addPrimitive(fn, 1, name=f"unary_{name}")
        for name, fn in self.operators.binary_operators.items():
            pset.addPrimitive(fn, 2, name=f"binary_{name}")

        # Remove default ARG* terminals from pset.terminals
        for ret_type, terms in list(pset.terminals.items()):
            filtered = [t for t in terms if not t.name.startswith("ARG")]
            pset.terminals[ret_type] = filtered

        # Also remove ARG* from context to avoid compile mapping
        for i in range(arity):
            pset.context.pop(f"ARG{i}", None)

        # Update arguments list so compile sees only variable names
        pset.arguments = [str(v) for v in sy2 + sy3]

        return pset

    def _variation(self, parents: list[gp.PrimitiveTree]) -> list[gp.PrimitiveTree]:
        """
        Apply crossover and mutation to parents to produce next-generation.
        Ensures that if there's an odd parent out, it's carried over unchanged.
        """
        offspring = []

        # iterate in pairs, preserving the final unpaired parent
        for a, b in zip_longest(parents[::2], parents[1::2], fillvalue=None):
            # clone to avoid overwriting originals in-place
            c1 = self.toolbox.clone(a)
            if b is None:
                offspring.append(c1)
                continue
            c2 = self.toolbox.clone(b)

            # 1‑point crossover
            if torch.rand(1).item() < self.cxpb:
                c1, c2 = self.toolbox.mate(c1, c2)

            if torch.rand(1).item() < self.mutpb:
                c1, = self.toolbox.mutate(c1)
            if torch.rand(1).item() < self.mutpb:
                c2, = self.toolbox.mutate(c2)

            # check if mutations are valid else get back to clone

            if not self.validate(c1):
                c1 = self.toolbox.clone(a)
            if not self.validate(c2):
                c2 = self.toolbox.clone(b)

            offspring.extend((c1, c2))

        return offspring

    def score(self):

        return self.best_fit

    def summary(self, pretty_print=True):

        if not self.fitted:
            warnings.warn("Model not fitted yet")
            return

        print(f'Best depth={self.best_depth}', end='\n\n')

        if pretty_print:
            # Print all expressions and loss per depth
            for depth, (fitness, expr, ind) in self.best_per_depth.items():
                """print(f"Depth={depth} fitness={fitness:.2f} expr:", end='\n\n')
                display(Math(sp.latex(sp.sympify(round_ind(expr), locals=self.operators.map))))
                print('\n\n')"""

                sym_expr = sp.sympify(expr, locals=self.operators.map)
                sym_expr = _round_floats(sym_expr)
                sym_expr = sym_expr.evalf(2)
                if self.labels is not None:
                    tex = sp.latex(sym_expr.subs(self.subs))
                else:
                    tex = sp.latex(sym_expr)
                display(Math(f"\\text{{Depth={depth} (fit={fitness:.2f})}}\\quad {tex}"))
                print('')
        else:
            for depth, (fitness, expr, ind) in self.best_per_depth.items():
                print(f'Depth={depth} fitness={fitness:.2f} expr: {expr}', end='\n\n')
