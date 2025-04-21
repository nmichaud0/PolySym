import unittest
from polysym import Regressor
import torch

class TestRegressor(unittest.TestCase):
    def setUp(self):
        # Example data for testing

        torch.manual_seed(42)

        n_obs = 50

        X3d = torch.zeros((n_obs, 2, 100))
        X2d = torch.zeros((n_obs, 1))
        y1d = torch.zeros(n_obs)
        y2d = torch.zeros(n_obs, 100)

        for obs in range(n_obs):

            start, end = torch.rand(2) * 100

            x1 = torch.linspace(start, end, 100)
            x2 = torch.cos(torch.linspace(start, end, 100))
            b = torch.randint(low=-10, high=10, size=(1, 1))

            y = (torch.cos(x1) + ((x1 * x2/ 10))) + b

            X3d[obs, 0] = x1
            X3d[obs, 1] = x2
            X2d[obs] = b
            y1d[obs] = torch.mean(y).item()
            y2d[obs] = y

        self.regressor2d = Regressor(X3d=X3d,
                                     X2d=X2d,
                                     y=y2d,
                                     max_complexity=10,
                                     pop_size=10,
                                     max_iter=20,
                                     seed=42,
                                     verbose=1,
                                     workers=1)

        self.regressor1d = Regressor(X3d=X3d,
                                     X2d=X2d,
                                     y=y1d,
                                     max_complexity=10,
                                     pop_size=10,
                                     max_iter=20,
                                     seed=42,
                                     verbose=1,
                                     workers=1)

    def test_fit_regressor1d(self):
        # Test fitting regressor1d does not raise errors
        try:
            self.regressor1d.fit()
        except Exception as e:
            self.fail(f"regressor1d.fit() raised {e}")

    def test_fit_regressor2d(self):
        # Test fitting regressor2d does not raise errors
        try:
            self.regressor2d.fit()
        except Exception as e:
            self.fail(f"regressor2d.fit() raised {e}")

    def test_predict_regressor1d_shape(self):
        # Test prediction shape for regressor1d
        self.regressor1d.fit()
        preds = self.regressor1d.predict()
        self.assertEqual(preds.shape, self.regressor1d.y.shape)

    def test_predict_regressor2d_shape(self):
        # Test prediction shape for regressor2d
        self.regressor2d.fit()
        preds = self.regressor2d.predict()
        self.assertEqual(preds.shape, self.regressor2d.y.shape)

    def test_score_regressor1d(self):
        # Test scoring for regressor1d
        self.regressor1d.fit()
        score = self.regressor1d.score()
        self.assertIsInstance(score, float)

    def test_score_regressor2d(self):
        # Test scoring for regressor2d
        self.regressor2d.fit()
        score = self.regressor2d.score()
        self.assertIsInstance(score, float)

    def test_predict_without_fit_regressor1d(self):
        # Test predict raises error if not fitted (regressor1d)
        reg = Regressor(X3d=self.regressor1d.X3d, X2d=self.regressor1d.X2d, y=self.regressor1d.y)
        with self.assertRaises(Exception):
            reg.predict()

    def test_predict_without_fit_regressor2d(self):
        # Test predict raises error if not fitted (regressor2d)
        reg = Regressor(X3d=self.regressor2d.X3d, X2d=self.regressor2d.X2d, y=self.regressor2d.y)
        with self.assertRaises(Exception):
            reg.predict()

if __name__ == '__main__':
    unittest.main()