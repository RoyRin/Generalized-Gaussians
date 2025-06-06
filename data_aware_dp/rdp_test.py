import unittest
from data_aware_dp import rdp

# write tests for rdp


class TestRDP(unittest.TestCase):
    # test get_SBM_RDP_interpolators

    def test_get_SBM_RDP_interpolators(self):
        beta = 1.0
        DEFAULT_Q_VALUE = 4096 / 45000
        interpolators = rdp.get_SBM_RDP_interpolators(beta, q=DEFAULT_Q_VALUE)
        pass
