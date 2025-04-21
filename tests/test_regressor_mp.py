import unittest
import torch
from multiprocessing import cpu_count
from polysym import Regressor


class TestRegressorMultiprocessing(unittest.TestCase):
    """Same tests as TestRegressor but using multiprocessing with half the CPU count."""

    def setUp(self):
        torch.manual_seed(42)
        n_obs, timesteps = 50, 100

        # prepare data
        X3d = torch.zeros((n_obs, 2, timesteps))
        X2d = torch.zeros((n_obs, 1))
        y1d = torch.zeros(n_obs)
        y2d = torch.zeros(n_obs, timesteps)

        for obs in range(n_obs):
            start, end = torch.rand(2) * 100
            x1 = torch.linspace(start, end, timesteps)
            x2 = torch.cos(torch.linspace(start, end, timesteps))
            b = torch.randint(low=-10, high=10, size=(1, 1))
            y = (torch.cos(x1) + (x1 * x2 / 10)) + b

            X3d[obs, 0] = x1
            X3d[obs, 1] = x2
            X2d[obs] = b
            y1d[obs] = torch.mean(y).item()
            y2d[obs] = y

        half_cpus = max(cpu_count() // 2, 1)
        self.regressor1d_mp = Regressor(
            X3d=X3d, X2d=X2d, y=y1d,
            max_complexity=10,
            pop_size=10,
            max_iter=20,
            seed=42,
            verbose=1,
            workers=half_cpus
        )
        self.regressor2d_mp = Regressor(
            X3d=X3d, X2d=X2d, y=y2d,
            max_complexity=10,
            pop_size=10,
            max_iter=20,
            seed=42,
            verbose=1,
            workers=half_cpus
        )

    def test_fit_regressor1d_mp(self):
        try:
            self.regressor1d_mp.fit()
        except Exception as e:
            self.fail(f"regressor1d_mp.fit() raised {e}")

    def test_fit_regressor2d_mp(self):
        try:
            self.regressor2d_mp.fit()
        except Exception as e:
            self.fail(f"regressor2d_mp.fit() raised {e}")

    def test_predict_regressor1d_shape_mp(self):
        self.regressor1d_mp.fit()
        preds = self.regressor1d_mp.predict()
        self.assertEqual(preds.shape, self.regressor1d_mp.y.shape)

    def test_predict_regressor2d_shape_mp(self):
        self.regressor2d_mp.fit()
        preds = self.regressor2d_mp.predict()
        self.assertEqual(preds.shape, self.regressor2d_mp.y.shape)

    def test_score_regressor1d_mp(self):
        self.regressor1d_mp.fit()
        score = self.regressor1d_mp.score()
        self.assertIsInstance(score, float)

    def test_score_regressor2d_mp(self):
        self.regressor2d_mp.fit()
        score = self.regressor2d_mp.score()
        self.assertIsInstance(score, float)

    def test_predict_without_fit_regressor1d_mp(self):
        reg = Regressor(X3d=self.regressor1d_mp.X3d,
                        X2d=self.regressor1d_mp.X2d,
                        y=self.regressor1d_mp.y)
        with self.assertRaises(Exception):
            reg.predict()

    def test_predict_without_fit_regressor2d_mp(self):
        reg = Regressor(X3d=self.regressor2d_mp.X3d,
                        X2d=self.regressor2d_mp.X2d,
                        y=self.regressor2d_mp.y)
        with self.assertRaises(Exception):
            reg.predict()


if __name__ == '__main__':
    unittest.main()
