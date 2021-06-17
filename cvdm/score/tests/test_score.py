import numpy as np
import numpy.testing as npt

from cvdm.score import score, Score
from cvdm.score.score import _so, _w

def test_so():
    tmp = _so(35, -29.8, 6.36)
    npt.assert_almost_equal(tmp, 0.9992446762628607, decimal=10)

    
def test_w():
    tmp = _w(3.62, 160, 1, np.array([0.71, 0.24, 0.018]))
    npt.assert_almost_equal(tmp, 0.8588, decimal=4)

                            
def test_score():
    #tmp = score(True, 55, 140, 160, True, True)
    tmp = score(True, 55, 3.62, 160, True, True)
    npt.assert_almost_equal(tmp, 0.02, decimal=2)
    #tmp = score(False, 60, 200, 160, True, True)
    tmp = score(False, 60, 5.17, 160, True, True)
    npt.assert_almost_equal(tmp, 0.08, decimal=2)


def test_score_json():
    sc = Score(low_risk=True)
    tmp = sc.score({"female":True,
                    "index_age": 55,
                    "chol_tot_mmol": 3.62,
                    "sbp": 160,
                    "cur_smoke": True})
    npt.assert_almost_equal(tmp, 0.02, decimal=2)
    tmp = sc.score({"female": False,
                    "index_age": 60,
                    "chol_tot_mmol": 5.17,
                    "sbp": 160,
                    "cur_smoke": True})
    npt.assert_almost_equal(tmp, 0.08, decimal=2)
