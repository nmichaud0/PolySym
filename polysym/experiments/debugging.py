import torch
# from PolySym import Regressor, Operators
from polysym.torch_operators import Operators
from polysym.regressor import Configurator
from polysym.model import PolySymModel
from polysym.evaluation import r2

n_obs = 1000

X3d = torch.zeros((n_obs, 2, 100))
X2d = torch.zeros((n_obs, 1))
y1d = torch.zeros(n_obs)
y2d = torch.zeros(n_obs, 100)

for obs in range(n_obs):

    start, end = torch.rand(2) * 100

    x1 = torch.linspace(start, end, 100)
    x2 = torch.cos(torch.linspace(start, end, 100))
    b = torch.randint(low=-10, high=10, size=(1, 1))

    y = (b + x1 + (x1 * x2)) * 14.31
    # expr=binary_add(binary_add(v0, x0), binary_mul(v1, v0))

    X3d[obs, 0] = x1
    X3d[obs, 1] = x2
    X2d[obs] = b
    y1d[obs] = torch.mean(y).item()
    y2d[obs] = y

operators = Operators(['add', 'sub', 'mul', 'div', 'neg'])

model = PolySymModel(X3d=X3d,
                        X2d=X2d,
                        y=y2d,
                        operators=operators,
                        min_complexity=4,
                        max_complexity=6,
                        pop_size=300,
                        stopping_criterion=.9,
                        max_iter=5,
                        fitness_fn = r2,
                        fitness_obj = 1,
                        seed=42,
                        verbose=1,
                        workers=1)

model.fit()
