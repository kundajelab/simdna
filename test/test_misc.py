import unittest
import simdna
from simdna import synthetic as sn
import numpy as np
from simdna import random
from collections import defaultdict

class TestBasics(unittest.TestCase):

    def test_direct_pwm_construction(self):
        pwm_rows = np.array([[0.095290, 0.318729, 0.083242, 0.502738],
                         [0.182913, 0.158817, 0.453450, 0.204819],
                         [0.307777, 0.053669, 0.491785, 0.146769],
                         [0.061336, 0.876232, 0.023001, 0.039430],
                         [0.008762, 0.989047, 0.000000, 0.002191],
                         [0.814896, 0.014239, 0.071194, 0.099671],
                         [0.043812, 0.578313, 0.365827, 0.012048],
                         [0.117325, 0.474781, 0.052632, 0.355263],
                         [0.933114, 0.012061, 0.035088, 0.019737],
                         [0.005488, 0.000000, 0.991218, 0.003293],
                         [0.365532, 0.003293, 0.621295, 0.009879],
                         [0.059276, 0.013172, 0.553238, 0.374314],
                         [0.013187, 0.000000, 0.978022, 0.008791],
                         [0.061538, 0.008791, 0.851648, 0.078022],
                         [0.114411, 0.806381, 0.005501, 0.073707],
                         [0.409241, 0.014301, 0.557756, 0.018702],
                         [0.090308, 0.530837, 0.338106, 0.040749],
                         [0.128855, 0.354626, 0.080396, 0.436123],
                         [0.442731, 0.199339, 0.292952, 0.064978]])

        pseudocount_prob=0.001
        pwm = simdna.pwm.PWM(name="some_name",
                             probMatrix=pwm_rows,
                             pseudocountProb=pseudocount_prob)

        pwm_rows = pwm_rows*(1-pseudocount_prob) + pseudocount_prob/4
        np.testing.assert_almost_equal(pwm_rows, np.array(pwm.getRows())) 
