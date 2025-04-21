import unittest
import torch
from polysym.torch_operators import (
    Operators,
    _unary_operators_map as UNARY_MAP,
    _binary_operators_map as BINARY_MAP,
    maybe_unsqueeze,
)

class BaseOperatorsTest(unittest.TestCase):
    """Common setup for testing unified Operators"""

    def setUp(self):
        torch.manual_seed(0)
        self.n = 20
        self.t = 10

        # Scalars
        self.s1 = 3.0
        self.s2 = -2.0
        self.zero = 0.0

        # 1‑D batches [n]
        self.batch1 = torch.randn(self.n)
        self.batch2 = torch.randn(self.n)
        self.zero_batch = torch.zeros(self.n)

        # 2‑D batches [n,t]
        self.vec1 = torch.randn(self.n, self.t)
        self.vec2 = torch.randn(self.n, self.t)
        self.zero_vec = torch.zeros(self.n, self.t)


class TestOperators(BaseOperatorsTest):
    """Test the unified Operators class"""

    def setUp(self):
        super().setUp()
        self.ops = Operators(select_all=True)

    def test_initialization(self):
        # all operators
        ops_all = Operators(select_all=True)
        self.assertEqual(set(ops_all.unary_operators), set(UNARY_MAP))
        self.assertEqual(set(ops_all.binary_operators), set(BINARY_MAP))

        # subset
        subset = ['add', 'sin', 'mean']
        ops_sub = Operators(operators_selection=subset)
        self.assertEqual(set(ops_sub.unary_operators), set(k for k in UNARY_MAP if k in subset))
        self.assertEqual(set(ops_sub.binary_operators), set(k for k in BINARY_MAP if k in subset))

    def test_unary_tensor_types(self):
        """Every unary op must accept scalar, 1‑D and 2‑D inputs."""
        for name, fn in UNARY_MAP.items():
            # scalar python
            out = fn(self.s1)
            self.assertIsInstance(out, torch.Tensor)

            # scalar tensor
            out = fn(torch.tensor(self.s2))
            self.assertIsInstance(out, torch.Tensor)

            # batch [n]
            out = fn(self.batch1)
            self.assertIsInstance(out, torch.Tensor)
            self.assertEqual(out.shape, self.batch1.shape if name not in ('mean','sum','std','min','max','median')
                             else ())  # reductions produce scalar for 1‑D

            # vector [n,t]
            out = fn(self.vec1)
            self.assertIsInstance(out, torch.Tensor)
            if name in ('mean','sum','std','min','max','median'):
                self.assertEqual(out.shape, (self.n,))
            else:
                self.assertEqual(out.shape, self.vec1.shape)

    def test_binary_tensor_types(self):
        """Every binary op must accept all shape combos and broadcast correctly."""
        for name, fn in BINARY_MAP.items():
            # two scalars
            out = fn(self.s1, self.s2)
            self.assertIsInstance(out, torch.Tensor)

            # scalar + tensor batch
            out = fn(self.s1, self.batch1)
            self.assertEqual(out.shape, self.batch1.shape)
            out = fn(self.batch1, self.s2)
            self.assertEqual(out.shape, self.batch1.shape)

            # tensor + tensor same shape
            out = fn(self.batch1, self.batch2)
            self.assertEqual(out.shape, self.batch1.shape)

            # division by zero should not throw
            if name in ('div',):
                out = fn(self.batch1, self.zero_batch)
                self.assertEqual(out.shape, self.batch1.shape)

            # 2‑D + 2‑D
            out = fn(self.vec1, self.vec2)
            self.assertEqual(out.shape, self.vec1.shape)

    def test_maybe_unsqueeze(self):
        """Directly test the broadcasting helper."""
        # 1‑D vs 2‑D
        a = torch.arange(5)         # [5]
        b = torch.randn(5,4)        # [5,4]
        a2, b2 = maybe_unsqueeze(a, b)
        self.assertEqual(a2.shape, (5,1))
        self.assertEqual(b2.shape, (5,4))

        # reverse
        a2, b2 = maybe_unsqueeze(b, a)
        self.assertEqual(a2.shape, (5,4))
        self.assertEqual(b2.shape, (5,1))

        # two scalars → each becomes 0‑D tensor
        x, y = maybe_unsqueeze(2.5, -1.3)
        self.assertTrue(x.dim()==0 and y.dim()==0)

    def test_dimension_mismatch_resolution(self):
        """
        For cases like dividing a reduced [n] by a [n,t] vector,
        unified operators must broadcast per‑observation.
        """
        # reduce vec1 → [n]
        min_res = self.ops.unary_operators['min'](self.vec1)
        self.assertEqual(min_res.shape, (self.n,))

        # power → [n,t]
        pow_res = self.ops.binary_operators['pow'](self.vec2, self.vec1)
        self.assertEqual(pow_res.shape, (self.n, self.t))

        # now divide → should broadcast [n] → [n,t]
        div_res = self.ops.binary_operators['div'](min_res, pow_res)
        self.assertEqual(div_res.shape, (self.n, self.t))

        # check correctness for one observation
        i = 2
        expected = min_res[i] / (pow_res[i] + torch.sign(pow_res[i])*1e-10)
        torch.testing.assert_close(div_res[i], expected)

if __name__ == '__main__':
    unittest.main()
