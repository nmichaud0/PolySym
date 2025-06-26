import copy

from polysym.regressor import Configurator
import torch
from torch import Tensor
import warnings
from polysym.utils import _evaluate_worker, unscale_expression, compile_tree2
from polysym.utils_expr import parse_to_tree
from polysym.dummy_mp import get_pool
import numpy as np
import random
from deap import tools, gp
import sympy as sp
import matplotlib.pyplot as plt
from math import isnan


class PolySymModel(Configurator):
    def __init__(self, X3d: Tensor, X2d: Tensor, y: Tensor, **kwargs):
        super().__init__(X3d, X2d, y, **kwargs)

        self.losses = []
        self.differential_losses = {depth+1: [] for depth in range(self.max_depth)}
        self.depth_stats = {depth+1: [] for depth in range(self.max_depth)}
        self.pop_refresh = 0

        self.cache = {}

    def fit(self):

        # pop = self.toolbox.population(n=self.pop_size)
        pop = self._balanced_population(self.pop_size)

        with get_pool(self.workers) as pool:
            # Cache single-individual evaluations
            def eval_cache(ind):
                key = ind.tree_str if hasattr(ind, 'tree_str') else str(ind)
                if key in self.cache:
                    return self.cache[key]
                res = _evaluate_worker(
                    ind,
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
                )
                self.cache[key] = res
                return res

            def eval_fitnesses(pop):
                # Prepare results list
                results = [None] * len(pop)
                # Identify individuals needing evaluation
                to_eval = []
                idx_map = {}
                for idx, ind in enumerate(pop):
                    key = ind.tree_str if hasattr(ind, 'tree_str') else str(ind)
                    if key in self.cache:
                        results[idx] = self.cache[key]
                    else:
                        idx_map[len(to_eval)] = idx
                        to_eval.append(ind)

                #print(f'New individuals proportion = {len(to_eval)}/{len(pop)}')
                #if len(pop) != self.pop_size:
                #    print(f'Evaluation size: {len(pop)} ; expected size = {self.pop_size}')

                # Evaluate uncached individuals
                if to_eval:
                    if self.workers == 1:
                        out = [eval_cache(ind) for ind in to_eval]
                    else:
                        args = [
                            (ind,
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
                             self.opt_sigma)
                            for ind in to_eval
                        ]
                        out = pool.starmap(_evaluate_worker, args)

                    # store in cache
                    for ind, r in zip(to_eval, out):
                        key = ind.tree_str if hasattr(ind, 'tree_str') else str(ind)
                        self.cache[key] = r

                    # Fill results in original order
                    for i, res in enumerate(out):
                        results[idx_map[i]] = res

                # Unpack final results
                fitnesses_ = [r[0] for r in results]
                mismatches = [r[1] for r in results]
                eph_flags_ = [r[2] for r in results]
                new_pop = [r[3] for r in results]
                return fitnesses_, mismatches, eph_flags_, new_pop

            ind_dim_mismatch = True
            max_iter = self.max_iter

            ### GENERATIONS ###
            while ind_dim_mismatch:

                for i in range(max_iter):

                    self.pop_refresh += 1

                    ## CHANGE MUTATION/CROSSOVER PROBS each 10 iter
                    # if lasts 50 gens didn't improve --> increase self.cxpb & self.mutpb by 10% of their own value
                    if i > 50 and i%50==0 and self.losses[-1] == self.losses[-10]:
                        cxpb = min(1., self.cxpb + .05)
                        mutpb = min(1., self.mutpb + .05)

                        if cxpb != self.cxpb or mutpb != self.mutpb:
                            self.logger.info(f'No improvements in loss for 50 generations; modifying genetic hyperparameters:')

                        if cxpb > self.cxpb:
                            self.logger.info(f'Increased mating probability by 5%: cxpb={cxpb:.2f}')
                            self.cxpb = cxpb
                        if mutpb > self.mutpb:
                            self.logger.info(f'Increased mutation probability by 5%: mutpb={mutpb:.2f}')
                            self.mutpb = mutpb

                    # NEW GEN
                    if i % 10 == 0:

                        bf = self.hof[0].fitness if len(self.hof.items) else self.worst_fitness

                        self.logger.info(f'Running iteration {i}/{max_iter}; best fit={bf:.2f}')

                    fitnesses, dim_mismatches, eph_flags, new_inds = eval_fitnesses(pop)

                    nan_fitnesses = []
                    # UPDATE INDIVIDUALS
                    for j, ind in enumerate(pop):
                        if isnan(fitnesses[j]):
                            nan_fitnesses.append(j)
                            continue
                        new_ind = new_inds[j]
                        pop[j] = copy.deepcopy(new_ind)
                        pop[j].dim_mismatch = dim_mismatches[j]
                        pop[j].fitness = fitnesses[j]
                        ind = pop[j]

                        ## UPDATE BESTS
                        if not ind.dim_mismatch:
                            depth = ind.height

                            old_fit, _, old_ind = self.best_per_depth[depth]
                            fitter = self.compare_func(ind.fitness, old_fit)
                            same_fit = ind.fitness == old_fit
                            shorter = old_ind is not None and len(ind) < len(old_ind)

                            # assign at right depth and minimize count of tree nodes
                            if fitter or (same_fit and shorter):
                                self.best_per_depth[depth] = (ind.fitness, str(ind), copy.deepcopy(ind))

                        else:
                            print(f'Found mismatch: {str(ind)} at gen {i}')

                        # CHECK BEST PER DEPTH DEBUG
                        """for depth, (fit_, expr_s, ind_) in self.best_per_depth.items():
                            if expr_s is None:
                                continue
                            if depth != parse_to_tree(expr_s, self.pset).height:
                                print(self.best_per_depth)
                                print(ind)
                                print(j)
                                print(str(ind))
                                raise ValueError('Found mismatch in best_per_depth')"""

                    pop = [pop[i] for i in range(len(pop)) if i not in nan_fitnesses]
                    fitnesses = [fitnesses[i] for i in range(len(fitnesses)) if i not in nan_fitnesses]
                    #print(f'Found {(sum(dim_mismatches)/self.pop_size)*100:.2f}% of nan mismatch')

                    self.update_differential_loss(pop)

                    best_fit = self.best_func(fitnesses)
                    self.losses.append(self.best_func(self.losses + [best_fit]))

                    finite = np.asarray(fitnesses)[np.isfinite(fitnesses)]

                    # All same, no need to change threshold
                    if min(fitnesses) != max(fitnesses):
                        self.threshold = np.percentile(finite, self.percentile)

                    self.hof.update(pop)

                    # ONLY EXPERIMENTAL:
                    # TODO: REMOVE
                    if max(fitnesses) == 1:
                        print('FOUND PERFECT R2... STOPPING ITERATIONS')
                        break

                    # GENERATE NEW POPULATION

                    # If no improvements over lasts 50 generations:
                    if i > 100 and self.losses[-1] == self.losses[-100] and self.pop_refresh >= 100:
                        # keep 10 best hof, fill half of pop size with
                        # fill last half with new pop
                        # make variations
                        best_hof = self.hof[:10] * int(self.pop_size // 20)
                        # new_pop = self.toolbox.population(n=int(self.pop_size // 2))
                        new_pop = self._balanced_population(self.pop_size // 2)
                        var_pop = best_hof + new_pop
                        random.shuffle(var_pop)
                        offsprings = self._variation(var_pop)
                        random.shuffle(offsprings)
                        pop =  offsprings[:self.pop_size]
                        self.logger.info(f'No improvements for 100 generations – population refresh')
                        self.pop_refresh = 0
                        # TODO: Fix stuff here (why keep hof at each gen ?)
                        # TODO: Every new generation must only contain new individuals // check if it messes with variations ?

                    else:
                        # 10% of hof size max + 10% of new population
                        # hof_size_p10 = int(self.hof_size*.1)
                        # pop_size_p90 = int(self.pop_size*1.1) # small overshoot to keep pop size identical

                        parents = tools.selTournament(pop, k=int(self.pop_size*.5) , tournsize=self.tournament_size)
                        # parents = tools.selTournament(pop, k=self.pop_size, tournsize=self.tournament_size)

                        # Get 10% of all best individuals to make variations on
                        # parents += 2 * self.hof.items[:hof_size_p10]
                        random.shuffle(parents)

                        offsprings = self._variation(parents + self.hof[:len(self.hof)//2])

                        #unique_novels = [o for o in offsprings if o not in self.cache]
                        #deficit = self.pop_size - len(unique_novels)

                        # add new individuals
                        # new_pop = self.toolbox.population(n=int(self.pop_size * .1))
                        new_pop = self._balanced_population(int(self.pop_size * .52))
                        # new_pop = self._balanced_population(deficit)

                        pop = (offsprings + new_pop)
                        random.shuffle(pop)
                        # pop = pop[:self.pop_size]  # fix overshoot

                        pop = self._rebalance2(pop)

                # remove all ind that have dimensionality mismatch
                clean_pop = [ind for ind in pop if ind.dim_mismatch is False] + self.hof.items

                if len(clean_pop) == 0:
                    max_iter = self.rerun_iter
                    print(
                        f"Did not find any individual that doesn't produce a dimensionality mismatch, rerunning simulation for {self.rerun_iter} iterations")
                else:
                    ind_dim_mismatch = False

            ### FINISHED TRAINING ###

        self.logger.info('Finished iterating, wrapping up fitting...')

        best_item = self.hof[0]
        self.best_fit = best_item.fitness

        # Get the lowest depth
        self.best_depth = [k for k, v in self.best_per_depth.items() if v[0] == self.best_fit][0]

        self.fitted = True

        # Unscaling
        if self.scale:
            self.unscale_expr()

        self.logger.info(f'Overall best depth={best_item.height} fitness={self.best_fit:.2f} ; expr={str(best_item)}')


    def unscale_expr(self):

        for depth, (fitness, expr, ind) in self.best_per_depth.items():

            expr = sp.sympify(expr)

            n_expr = unscale_expression(expr, self.norm_stats)
            self.best_per_depth[depth] = (fitness, n_expr, ind)


    def eval_expr(self, expr: str):

        tree = parse_to_tree(expr, self.pset)
        func = compile_tree2(tree, self.pset)

        raw = func(*self.inputs)

        return self.fitness_fn(raw, self.y)

    def get_expr(self, expr: str):
        return parse_to_tree(expr, self.pset)

    def predict_expr(self, expr: str):

        tree = parse_to_tree(expr, self.pset)
        func = compile_tree2(tree, self.pset)
        raw = func(*self.inputs)
        return raw

    def update_differential_loss(self, pop):

        best_per_depth = {d: [] for d in range(1, self.max_depth + 1)}
        pop_count = {d: 0 for d in range(1, self.max_depth + 1)}

        for ind in pop:
            d = ind.height
            pop_count[d] += 1

            # scalar fitness value
            fit = float(ind.fitness if isinstance(ind.fitness, (int, float))
                        else ind.fitness.values[0])

            if np.isfinite(fit):
                best_per_depth[d].append(fit)

        # ------- update curves ----------------------------------------
        for d in range(1, self.max_depth + 1):
            # best value this generation or worst_fitness sentinel
            cur_best = (self.best_func(best_per_depth[d])
                        if best_per_depth[d] else self.worst_fitness)

            if self.differential_losses[d]:
                prev = self.differential_losses[d][-1]
                self.differential_losses[d].append(self.best_func((cur_best, prev)))
            else:
                self.differential_losses[d].append(cur_best)

            # store population count
            self.depth_stats[d].append(pop_count[d])


    def plot_differential_loss(self):

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.set_xlabel('Generation', fontsize=15)
        ax.set_ylabel(f'Loss: {self.fitness_fn.__name__.capitalize()}', fontsize=15)
        ax.set_ylim(-1, 1)

        for depth, fitnesses in self.differential_losses.items():
            ax.plot(fitnesses, label=f'Depth={depth}', lw=2)

        ax.legend(loc='lower right')
        fig.suptitle('PolySym: Fitness function per depth per generation', fontsize=18)
        plt.show()

    def plot_depth_per_gen(self):

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.set_xlabel('Generation', fontsize=15)
        ax.set_ylabel(f'Population count', fontsize=15)

        for depth, n_ind in self.depth_stats.items():
            ax.plot(n_ind, label=f'Depth={depth}', lw=2)

        ax.legend(loc='lower right')
        fig.suptitle('PolySym: Population per depth per generation', fontsize=18)
        plt.show()
