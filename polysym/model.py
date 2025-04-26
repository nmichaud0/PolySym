from polysym.regressor import Configurator
import torch
from torch import Tensor
import warnings
from polysym.utils import _evaluate_worker, unscale_expression, compile_tree
from polysym.dummy_mp import get_pool
import numpy as np
import random
from deap import tools, gp
import sympy as sp
import matplotlib.pyplot as plt


class PolySymModel(Configurator):
    def __init__(self, X3d: Tensor, X2d: Tensor, y: Tensor, **kwargs):
        super().__init__(X3d, X2d, y, **kwargs)

        self.losses = []
        self.differential_losses = {depth+1: [] for depth in range(self.max_depth)}
        self.depth_stats = {depth+1: [] for depth in range(self.max_depth)}
        self.pop_refresh = 0


    def fit(self):

        # pop = self.toolbox.population(n=self.pop_size)
        pop = self._balanced_population(self.pop_size)

        with get_pool(self.workers) as pool:
            def eval_fitnesses(pop):

                if self.workers == 1:
                    result = [self.toolbox.evaluate(ind) for ind in pop]
                else:

                    args = [(ind,
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
                             self.ngsa2_alpha)
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

            ### GENERATIONS ###
            while ind_dim_mismatch:

                for i in range(max_iter):
                    self.pop_refresh += 1

                    ## CHANGE MUTATION/CROSSOVER PROBS each 10 iter
                    # if lasts 50 gens didn't improve --> increase self.cxpb & self.mutpb by 10% of their own value
                    if i > 50 and i%50==0 and self.losses[-1] == self.losses[-10]:
                        cxpb = min(1., self.cxpb + .05)
                        mutpb = min(1., self.mutpb + .05)

                        if cxpb > self.cxpb:
                            self.logger.info(f'Increased mating probability by 5%: cxpb={cxpb:.2f}')
                            self.cxpb = cxpb
                        if mutpb > self.mutpb:
                            self.logger.info(f'Increased mutation probability by 5%: mutpb={mutpb:.2f}')
                            self.mutpb = mutpb

                    # NEW GEN
                    if i % 10 == 0:

                        bf = self.hof.items[0].fitness if len(self.hof.items) else self.worst_fitness

                        self.logger.info(f'Running iteration {i}/{max_iter}; best fit={bf:.2f}')

                    fitnesses, dim_mismatches, eph_flags, new_inds = eval_fitnesses(pop)

                    # UPDATE INDIVIDUALS
                    for j, ind in enumerate(pop):

                        if eph_flags[j]:
                            new_ind = new_inds[j]
                            new_ind.dim_mismatch = dim_mismatches[j]
                            pop[j] = new_ind
                            ind = new_ind
                        else:
                            ind.dim_mismatch = dim_mismatches[j]

                        ind.fitness = fitnesses[j]

                        ## UPDATE BESTS
                        if not dim_mismatches[j]:
                            depth = ind.height

                            curr_fit, _, curr_ind = self.best_per_depth[depth]
                            fitter = self.compare_func(ind.fitness, curr_fit)
                            same_fit = ind.fitness == curr_fit
                            shorter = curr_ind is not None and len(ind) < len(curr_ind)

                            # assign at right depth and minimize count of tree nodes
                            if fitter or (same_fit and shorter):
                                self.best_per_depth[depth] = (ind.fitness, str(ind), ind)

                        else:
                            print(f'Found mismatch: {str(ind)}')

                    self.update_differential_loss(pop)

                    best_fit = self.best_func(fitnesses)
                    self.losses.append(self.best_func(self.losses + [best_fit]))

                    finite = np.asarray(fitnesses)[np.isfinite(fitnesses)]
                    self.threshold = np.percentile(finite, self.percentile)
                    self.hof.update(pop)

                    # CHECK IF STOP
                    # IF STOPPING CRITERION IS MET OVER ALL SEARCHED DEPTHS
                    # OR IF diff loss signal is met
                    
                    # Check if all depths from min to max have met stopping criterion
                    stop_criterion_met = False
                    if self.stopping_criterion:
                        all_depths_met = True
                        for depth in range(self.min_depth, self.max_depth + 1):
                            fit, _, _ = self.best_per_depth[depth]
                            if not self.compare_func(fit, self.stopping_criterion):
                                all_depths_met = False
                                break
                        stop_criterion_met = all_depths_met
                    
                    # Check if we should stop based on either criterion
                    if stop_criterion_met or self.diff_loss_signal():
                        ind_dim_mismatch = False
                        stop_reason = 'Stopping criterion met for all depths' if stop_criterion_met else 'Differential loss signal detected'
                        self.logger.info(f'{stop_reason}; stopping iterations')
                        break

                    if i == max_iter - 1:
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

                    else:
                        # 10% of hof size max + 10% of new population
                        hof_size_p10 = int(self.hof_size*.1)
                        pop_size_p90 = int(self.pop_size*.92) # small overshoot to keep pop size identical

                        parents = tools.selTournament(pop, k=pop_size_p90-hof_size_p10 , tournsize=self.tournament_size)

                        # Get 10% of all best individuals to make variations on
                        parents += 2 * self.hof.items[:hof_size_p10]
                        random.shuffle(parents)

                        offsprings = self._variation(parents)

                        # add new individuals
                        # new_pop = self.toolbox.population(n=int(self.pop_size * .1))
                        new_pop = self._balanced_population(int(self.pop_size * .1))

                        pop = (offsprings + new_pop)
                        random.shuffle(pop)
                        pop = pop[:self.pop_size]  # fix overshoot

                    pop = self._rebalance(pop)

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

        pop = clean_pop

        best_fit = self.worst_fitness
        best_depth = None
        best_ind = None

        for depth, (fitness, expr, ind) in self.best_per_depth.items():

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

        self.best_depth = best_depth
        self.best_expr = sp.sympify(str(best_ind))
        self.best_fit = self.best_func(fitnesses)

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

        # Unscaling
        if self.scale:
            self.unscale_expr()

        self.logger.info(f'Overall best depth={best_depth} fitness={best_fit:.2f} ; expr={self.best_expr}')

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


    def unscale_expr(self):

        for depth, (fitness, expr, ind) in self.best_per_depth.items():

            expr = sp.sympify(expr)

            n_expr = unscale_expression(expr, self.norm_stats)
            self.best_per_depth[depth] = (fitness, n_expr, ind)


    def eval_expr(self, expr: str):

        func = compile_tree(expr, self.pset)

        raw = func(*self.inputs)

        return self.fitness_fn(raw, self.y)

    def update_differential_loss(self, pop):

        ind_per_depth = {}
        count_per_depth = {depth+1: 0 for depth in range(self.max_depth)}
        for ind in pop:
            depth = ind.height
            if depth not in ind_per_depth:
                ind_per_depth[depth] = []

            count_per_depth[depth] += 1
            ind_per_depth[depth].append(ind.fitness)

        for depth, fitnesses in ind_per_depth.items():

            current_loss = self.best_func(fitnesses)
            if len(self.differential_losses[depth]) == 0:
                self.differential_losses[depth].append(current_loss)
            else:
                last_loss = self.differential_losses[depth][-1]
                self.differential_losses[depth].append(self.best_func((current_loss, last_loss)))

        for depth, count in count_per_depth.items():
            if count == 0:
                if len(self.differential_losses[depth]) == 0:
                    self.differential_losses[depth].append(self.worst_fitness)
                else:
                    self.differential_losses[depth].append(self.differential_losses[depth][-1])

            self.depth_stats[depth].append(count)


    def diff_loss_signal(self):
        """
        Tells whether diff losses haven't changed since a long time or enough individuals have been evaluated
        :return:
        """

        last_n_iter = 100
        last_n_ind = 1000
        
        depths = list(range(self.min_depth, self.max_depth+1))

        signals = {depth: False for depth in depths}

        for depth in depths:

            losses, count_per_gen = self.differential_losses[depth], self.depth_stats[depth]

            if len(losses) < last_n_iter:
                return False

            # Check if loss haven't changed since 100 iter
            # If yes, have there been at least 1000 individuals since last 100 iter ?
            # If yes, then signal is sent for that depth

            if self.compare_func(losses[-last_n_iter], losses[-1]):
                if sum(count_per_gen[-last_n_iter:]) >= last_n_ind:
                    signals[depth] = True

        return all(list(signals.values()))

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
