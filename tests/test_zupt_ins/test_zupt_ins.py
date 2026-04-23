import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import pytest
import src.zupt_ins.zupt_ins as zupt_ins
import numpy as np

def test_state_matrix():
    q = np.array([1.2,1.1,0.2,0.3], dtype=float)
    u = np.array([1.1,1.111,0.235,0.3, 1.2,1.9], dtype=float)
    Ts = 0.01

    F, G = zupt_ins.state_matrix(q, u, Ts)

    print(f"\n{F = }")
    print(f"\n{G = }")
