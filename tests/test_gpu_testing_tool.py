import unittest

import numpy as np

from gpu_testing_tool import normalize_shape, resolve_numpy_dtype


class NormalizeShapeTests(unittest.TestCase):
    def test_pad_to_target_shape(self):
        arr = np.arange(6, dtype=np.uint16).reshape(2, 3)
        out = normalize_shape(arr, [1, 1, 4, 5])
        self.assertEqual(out.shape, (1, 1, 4, 5))
        np.testing.assert_array_equal(out[0, 0, :2, :3], arr)
        self.assertTrue(np.all(out[0, 0, 2:, :] == 0))
        self.assertTrue(np.all(out[0, 0, :, 3:] == 0))

    def test_trim_if_larger_than_fixed_dim(self):
        arr = np.arange(24, dtype=np.uint16).reshape(1, 1, 4, 6)
        out = normalize_shape(arr, [1, 1, 3, 5])
        self.assertEqual(out.shape, (1, 1, 3, 5))
        np.testing.assert_array_equal(out, arr[:, :, :3, :5])


class DTypeTests(unittest.TestCase):
    def test_resolve_uint16(self):
        self.assertEqual(resolve_numpy_dtype("tensor(uint16)"), np.dtype(np.uint16))


if __name__ == "__main__":
    unittest.main()
