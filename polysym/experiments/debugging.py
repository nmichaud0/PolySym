import torch
# from PolySym import Regressor, Operators
from polysym.torch_operators_2 import Operators
from polysym.regressor import Configurator
from polysym.model import PolySymModel
from polysym.evaluation import r2
import random

n_obs = 10

X3d = torch.zeros((n_obs, 2, 50))
sequences = []
X2d = torch.zeros((n_obs, 1))
y1d = torch.zeros(n_obs)
# y2d = torch.zeros(n_obs, 150)
y2d = torch.full((n_obs, 50), torch.nan)

for obs in range(n_obs):

    length = random.randint(25, 50)
    if obs==0:
        length = 50
    start, end = torch.rand(2) * 100

    x1 = torch.linspace(start, end, length)
    x2 = torch.cos(torch.linspace(start, end, length))

    x12 = torch.vstack([x1, x2]).T
    sequences.append(x12)

    b = torch.randint(low=-10, high=10, size=(1, 1))

    y = (b + x1 + (x1 * x2)) + 14
    # expr=binary_add(binary_add(x0, n0), binary_mul(v1, v0))

    expr = 'binary_add(binary_add(x0, v0), binary_mul(v0, v1))'

    X3d[obs, 0, :length] = x1
    X3d[obs, 1, :length] = x2
    X2d[obs] = b
    y1d[obs] = torch.nanmean(y).item()
    y2d[obs, :length] = y

sequences = torch.nn.utils.rnn.pad_sequence(sequences, padding_value=torch.nan)
X3d = sequences
X3d = X3d.permute(1, 2, 0)
print(X3d.shape)
operators = Operators(['add', 'sub', 'mul', 'div', 'neg', 'mean'])

model = PolySymModel(X3d=X3d,
                        X2d=X2d,
                        y=y2d,
                        operators=None,
                        min_complexity=1,
                        max_complexity=4,
                        pop_size=200,
                        max_iter=200,
                        scale=False,
                        fitness_fn = r2,
                        fitness_obj = 1,
                        seed=43,
                        verbose=1,
                        optimize_ephemerals=True,
                        workers=1)

#model.eval_expr('add(add(x0, v0), mul(v0, v1))')


import multiprocessing as mp


mp.set_start_method('fork', force=True)

model.fit()

model.summary(pretty_print=False)