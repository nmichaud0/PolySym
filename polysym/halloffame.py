from deap import creator

class HallOfFame:
    def __init__(self, maxsize: int, objective: int):
        self.maxsize = maxsize
        self.objective = objective  # 1 higher=better ; -1 lower=better
        assert objective in (-1, 1), f'Objective must be in (-1, 1). Got: {self.objective=}'
        self.reverse = True if objective == 1 else False

        self.items = []

    def update(self, pop):

        self.items += pop
        self._update()

        """for ind in self.items:
            if ind.fitness >= 0 > ind.fitness:
                print(ind, str(ind))
                raise ValueError('Ind fitness is nan')"""

    @staticmethod
    def _scalar_fit(ind):

        if isinstance(ind.fitness, (float, int)):
            return ind.fitness
        return ind.fitness[0]

    def _update(self):

        #self.items = sorted(self.items, key=object.fitness, reverse=True)[:self.maxsize]

        self.items = sorted(self.items, key=lambda ind: self._scalar_fit(ind), reverse=self.reverse)[:self.maxsize]

    def __getitem__(self, item):
        return self.items[item]


