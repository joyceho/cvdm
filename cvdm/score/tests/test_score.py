import numpy.testing as npt

from cvdm.score import score, Score


def test_score():
    tmp = score(True, 55, 140, 160, True, True)
    npt.assert_almost_equal(tmp, 0.02, decimal=2)
    tmp = score(False, 60, 200, 160, True, True)
    npt.assert_almost_equal(tmp, 0.08, decimal=2)


def test_score_json():
    sc = Score(lowRisk=True)
    tmp = sc.score({"female":True,
                    "index_age": 55,
                    "chol_tot": 140,
                    "sbp": 160,
                    "cur_smoke": True})
    npt.assert_almost_equal(tmp, 0.02, decimal=2)
    tmp = sc.score({"female": False,
                    "index_age": 60,
                    "chol_tot": 200,
                    "sbp": 160,
                    "cur_smoke": True})
    npt.assert_almost_equal(tmp, 0.08, decimal=2)
