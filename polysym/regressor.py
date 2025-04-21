from polysym.evaluation import rmse, mse
from polysym.torch_operators import Operators
from polysym.utils import seed_everything, get_logger, _RandConst
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

# TODO: multi-objective regressor

def _hill_climb_constants(ind, inputs, pset, fitness_fn, y,
                          objective, worst_fitness,
                          steps: int = 20,
                          sigma: float = 0.1):
    """
    Very simple hill‑climbing over the ephemeral 'randc' constants in `ind`.
    Returns a list of new constant values of the same length/order.
    """
    # find where the ephemerals live
    eph = _extract_ephemerals(ind)
    if not eph:
        return []
    idxs, orig_vals = zip(*eph)
    best_vals = list(orig_vals)
    best_fit = worst_fitness
    best_ind = ind

    # evaluate a fresh copy with given vals
    def eval_with(vals):
        tmp = copy.deepcopy(ind)
        tmp = _apply_ephemerals(tmp, vals, eph)
        func = gp.compile(expr=tmp, pset=pset)
        out = func(*inputs)
        if out.dim() != objective:
            return worst_fitness, tmp
        return float(fitness_fn(out, y)), tmp

    # initialize
    fit0, tmp_ind = eval_with(best_vals)
    if fit0 < best_fit:
        best_fit = fit0
        best_ind = tmp_ind

    # simple gaussian perturbations
    for _ in range(steps):
        cand = [v + sigma * random.gauss(0,1) for v in best_vals]
        f, ind_tmp = eval_with(cand)
        if f < best_fit:
            best_fit, best_vals, best_ind = f, cand, ind_tmp

    return best_vals, best_ind


def _extract_ephemerals(ind: gp.PrimitiveTree):
    """
    Find all the ephemeral‐constant nodes (name 'randc') in the tree
    and return their (index, original_value) in tree order.
    """
    out = []
    for idx, node in enumerate(ind):
        # ephemeral constants are stored as gp.Terminal with name 'randc'
        if isinstance(node, gp.Terminal) and node.name == 'randc':
            out.append((idx, node.value))
    return out

def _apply_ephemerals(ind: gp.PrimitiveTree, new_values: list[float], extracted_ephemerals=None):
    """
    Overwrite only the actual ephemeral‐constant nodes ('randc')
    in the individual with the new_values.
    """
    if extracted_ephemerals is None:
        eph = _extract_ephemerals(ind)
    else:
        eph = extracted_ephemerals

    if len(eph) != len(new_values):
        raise ValueError(f"Expected {len(eph)} ephemerals, got {len(new_values)}")

    for ((idx, old_val), new_val) in zip(eph, new_values):
        node = ind[idx]
        # sanity check
        assert isinstance(node, gp.Terminal) and hasattr(node, "value"), f'Node as position {idx} is not an ephemeral constant'

        node.value = new_val

    return ind


def _evaluate_worker(ind, inputs, pset, fitness_fn, y, worst_fitness, objective, optimize_ephemerals):
    """
    Multiprocessing implementation

    returns fitness, dimensionality mismatch (whether tree doesn't return expected shape),
    and individual
    flag tells if ephemeral optimization was done

    """
    func = gp.compile(expr=ind, pset=pset)

    raw = func(*inputs)

    dimensionality_mismatch = raw.dim() != objective

    # Do we have ephemerals ?
    eph = bool([v for _, v in _extract_ephemerals(ind)])

    fitness = worst_fitness if dimensionality_mismatch else fitness_fn(raw, y)

    if not eph or not optimize_ephemerals:
        return fitness, dimensionality_mismatch, False, ind

    ephemerals, better_ind = _hill_climb_constants(ind, inputs, pset, fitness_fn, y, objective, worst_fitness)

    return fitness, dimensionality_mismatch, True, better_ind

# TODO: check labels and final expression rendering


class Regressor:
    def __init__(self, X3d: Tensor,
                       X2d: Tensor,
                       y: Tensor,
                       max_complexity: int = 10,
                       min_complexity: int = 1,
                       max_iter: int = 1000,
                       rerun_iter: int = 10,
                       stopping_criterion: Union[float, bool]=False,
                       labels_3d: list[str] = None,
                       labels_2d: list[str] = None,
                       min_max_constants: tuple[int, int] = (-100, 100),
                       optimize_ephemerals: bool = True,
                       operators: Operators = None,
                       fitness_fn: Callable = rmse,
                       fitness_obj: Literal[-1, 1] = -1,
                       pop_size: int = 100,
                       tournament_size: int = 3,
                       cxpb: float = .5,
                       mutpb: float = .2,
                       seed: Union[int, None] = None,
                       verbose: int = 0,
                       workers: int = -1
                   ):

        self.X3d = X3d  # shall be of shape (n_obs, variables, time)
        self.X2d = X2d  # shape (n_obs, variables)
        self.y = y

        self.max_depth = max_complexity
        self.min_depth = min_complexity

        assert self.max_depth >= self.min_depth, f'Minimum complexity must be of 2, got {self.max_depth}'

        self.max_iter = max_iter
        self.stopping_criterion = stopping_criterion
        if (not isinstance(self.stopping_criterion, float)) and (not (self.stopping_criterion is False)):
            raise ValueError(f"stoping criterion must be either float or False, got: {stopping_criterion}")

        self.rerun_iter = rerun_iter

        self.labels_3d = labels_3d
        self.labels_2d = labels_2d

        self.min_constant = min_max_constants[0]
        self.max_constant = min_max_constants[1]

        self.optimize_ephemerals = optimize_ephemerals

        self.fitness_fn = fitness_fn
        self.fitness_obj = fitness_obj
        self.worst_fitness = float('-inf') if fitness_obj == 1 else float('inf')
        self.pop_size = pop_size
        self.tournament_size = tournament_size
        self.cxpb = cxpb
        self.mutpb = mutpb
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

        for i in range(self.X2d.shape[1]):
            self.inputs.append(self.X2d[:, i])

        for i in range(self.X3d.shape[1]):
            self.inputs.append(self.X3d[:, i, :])


        self.pset = self._build_primitives()
        self.toolbox = self._setup_gp()

        # lower than if minimization objective else higher than
        self.compare_func = operator.lt if self.fitness_obj == -1 else operator.gt

        self.best_per_depth = {}
        self.best_per_depth = {depth+1: (self.worst_fitness, None) for depth in range(self.max_depth)}
        self.best_expr = None
        self.best_compiled = None
        self.best_fit = None
        self.fitted = False

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


    def fit(self):

        pop = self.toolbox.population(n=self.pop_size)

        pool = Pool(self.workers)


        def eval_fitnesses(pop):

            if self.workers == 1:
                result = [self.toolbox.evaluate(ind) for ind in pop]
            else:

                args = [(ind,
                         self.inputs,
                         self.pset,
                         self.fitness_fn,
                         self.y,
                         self.worst_fitness,
                         self.objective,
                         self.optimize_ephemerals)
                        for ind in pop]

                result = pool.starmap(_evaluate_worker, args)

            fitnesses_, mismatches, eph_flags_, new_pop = [], [], [], []
            for i in result:
                fitnesses_.append(i[0])
                mismatches.append(i[1])
                eph_flags_.append(i[2])
                new_pop.append(i[3])

            return fitnesses_, mismatches, eph_flags_, new_pop

        ind_dim_mismatch = True
        max_iter = self.max_iter
        while ind_dim_mismatch:

            for i in range(max_iter):

                if i % 10 == 0:
                    self.logger.info(f'Running iteration {i}/{max_iter}')

                fitnesses, dim_mismatches, eph_flags, new_inds = eval_fitnesses(pop)

                for j, ind in enumerate(pop):

                    if eph_flags[j]:
                        new_ind = new_inds[j]
                        new_ind.dim_mismatch = dim_mismatches[j]
                        new_ind.fitness = fitnesses[j]
                        pop[j] = new_ind
                        ind = new_ind
                    else:
                        ind.dim_mismatch = dim_mismatches[j]
                        ind.fitness = fitnesses[j]

                    ## update best individuals
                    if not dim_mismatches[j]:
                        depth = ind.height
                        try:
                            if self.compare_func(ind.fitness, self.best_per_depth[depth][0]):
                                self.best_per_depth[depth] = (ind.fitness, ind)
                        except Exception:
                            x000 = 0
                            print(ind.fitness, self.best_per_depth)


                if i == max_iter-1:
                    break

                parents = tools.selTournament(pop, k=self.pop_size, tournsize=self.tournament_size)
                pop = self._variation(parents)

            # remove all ind that have dimensionality mismatch
            clean_pop = [ind for ind in pop if ind.dim_mismatch is False]

            if len(clean_pop) == 0:
                max_iter = self.rerun_iter
                print(f"Did not find any individual that doesn't produce a dimensionality mismatch, rerunning simulation for {self.rerun_iter} iterations")
            else:
                ind_dim_mismatch = False
                pool.close()
                pool.join()

        self.logger.info('Finished iterating, wrapping up fitting...')

        pop = clean_pop

        best_fit = self.worst_fitness
        best_depth = None
        best_ind = None

        for depth, (fitness, ind) in self.best_per_depth.items():

            if depth is None:
                # Never produced valid for that depth
                continue

            better = self.compare_func(fitness, best_fit)

            if better:
                best_depth = depth
                best_ind = ind
                best_fit = fitness

        if best_ind is None:
            warnings.warn(f"Didn't find any good fit, must be an internal error")

        best_ind = pop[0]

        self.best_expr = sp.sympify(str(best_ind))
        self.best_fit = best_fit

        unary_map = self.operators.get_unary_dict()  # e.g. {'sin': fn, 'sqrt': fn, ...}
        binary_map = self.operators.get_binary_dict()  # e.g. {'add': fn, 'mul': fn, ...}

        primitives_ctx = {
            **{f'unary_{name}': fn for name, fn in unary_map.items()},
            **{f'binary_{name}': fn for name, fn in binary_map.items()},
        }

        self.best_compiled = sp.lambdify(self.symbols,
                                         self.best_expr,
                                         modules=[primitives_ctx, {'torch': torch}])
        self.fitted = True

        self.logger.info(f'Overall best depth={best_depth} fitness={best_fit} ; expr={self.best_expr}')

    def predict(self, X_3d: Tensor = None, X_2d: Tensor = None):
        """Make predictions using the best expression found during fitting.

        Args:
            X_3d: Optional 3D tensor for prediction. If None, uses the training data.
            X_2d: Optional 2D tensor for prediction. If None, uses the training data.

        Returns:
            Tensor: Predictions with the same shape as the target variable
        """
        assert self.fitted, 'Model not fitted'
        
        # If no new data is provided, use training data
        X_3d = self.X3d if X_3d is None else X_3d
        X_2d = self.X2d if X_2d is None else X_2d

        # Check that input data has the correct shape
        assert X_3d.shape[1:] == self.X3d.shape[1:], f"X_3d shape mismatch: {X_3d.shape[1:]} vs {self.X3d.shape[1:]}"
        assert X_2d.shape[1:] == self.X2d.shape[1:], f"X_2d shape mismatch: {X_2d.shape[1:]} vs {self.X2d.shape[1:]}"

        # Prepare inputs for the compiled expression
        n_samples = X_3d.shape[0]
        inputs = []

        # Flatten inputs to match how the model was trained
        for i in range(X_2d.shape[1]):
            inputs.append(X_2d[:, i])

        for i in range(X_3d.shape[1]):
            inputs.append(X_3d[:, i])

        # Use the compiled expression for prediction
        result = self.best_compiled(*inputs)

        if self.objective == 1:
            out = result if torch.is_tensor(result) else torch.tensor(result)
            return out.reshape(-1)
        else:
            return result if torch.is_tensor(result) else torch.tensor(result)
        
        

    def _setup_gp(self):

        if "Fitness" not in creator.__dict__:
            creator.create("Fitness", base.Fitness, weights=(float(self.fitness_obj),))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness, dim_mismatch=None)

        tb = base.Toolbox()
        tb.register('expr_init', gp.genFull, pset=self.pset, min_=self.min_depth, max_=self.max_depth)
        tb.register("individual", tools.initIterate, creator.Individual, tb.expr_init)
        tb.register("population", tools.initRepeat, list, tb.individual)

        tb.register('clone', copy.deepcopy)

        tb.register('mutate', gp.mutUniform, expr=tb.expr_init, pset=self.pset)

        def _evaluate(ind):

            func = gp.compile(expr=ind, pset=self.pset)

            self.logger.debug(f'Evaluating expression={str(ind)}')

            raw = func(*self.inputs)

            raw_dim = raw.dim()

            dimensionality_mismatch = raw_dim != self.objective

            # Do we have ephemerals ?
            eph = bool([v for _, v in _extract_ephemerals(ind)])

            fitness = self.worst_fitness if dimensionality_mismatch else self.fitness_fn(raw, self.y)

            if not eph or not self.optimize_ephemerals:
                return fitness, dimensionality_mismatch, False, ind

            ephemerals, better_ind = _hill_climb_constants(ind,
                                               self.inputs,
                                               self.pset,
                                               self.fitness_fn,
                                               self.y,
                                               self.objective,
                                               self.worst_fitness)

            return fitness, dimensionality_mismatch, True, better_ind


        tb.register("evaluate", _evaluate)
        tb.register("select", tools.selTournament, tournsize=self.tournament_size)
        tb.register("mate", gp.cxOnePoint)

        tb.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=self.max_depth))
        tb.decorate('mutate', gp.staticLimit(key=operator.attrgetter('height'), max_value=self.max_depth))

        return tb
    
    def _build_primitives(self):
        """Create symbols and DEAP PrimitiveSet once."""
    
        n2, n3 = self.X2d.shape[1], self.X3d.shape[1]
    
        # Handle both cases: single symbol or multiple symbols
        if n2 == 1:
            sy2 = [sp.symbols("x0")]
        else:
            sy2 = list(sp.symbols(" ".join(f"x{i}" for i in range(n2))))
    
        if n3 == 1:
            sy3 = [sp.symbols("v0")]
        else:
            sy3 = list(sp.symbols(" ".join(f"v{i}" for i in range(n3))))
    
        self.symbols = sy2 + sy3
    
        pset = gp.PrimitiveSet("MAIN", len(self.symbols))
        pset.renameArguments(**{f"ARG{i}": str(s) for i, s in enumerate(self.symbols)})
    
        # Get function lists with their names
        unary_fns = list(self.operators.unary_operators.items())
        binary_fns = list(self.operators.binary_operators.items())
        
        # Add unary functions with their proper names
        for name, fn in unary_fns:
            pset.addPrimitive(fn, 1, name=f"unary_{name}")
    
        # Add binary functions with their proper names
        for name, fn in binary_fns:
            pset.addPrimitive(fn, 2, name=f"binary_{name}")
    
        for s in self.symbols:
            pset.addTerminal(s, name=str(s))

        randc = _RandConst(self.min_constant, self.max_constant)

        pset.addEphemeralConstant("randc", randc)
        
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

            offspring.extend((c1, c2))

        return offspring

    def score(self):

        return self.best_fit

    def summary(self):

        if not self.fitted:
            warnings.warn("Model not fitted yet")
            return

        print(f"Best depth={self.best_per_depth[1][0]} fitness={self.best_per_depth[1][0]} ; expr={self.best_expr}")

        # Print all expressions and loss per depth
        for depth, (fitness, ind) in self.best_per_depth.items():
            print(f"Depth={depth} fitness={fitness} ; expr={str(ind)}")