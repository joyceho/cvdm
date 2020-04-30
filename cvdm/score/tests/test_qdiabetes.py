import numpy.testing as npt

from cvdm.score import qdiabetes, QDiabetes


def test_qdiabetes():
    tmp = qdiabetes(64, False, 27.34, 0.5, False, False,
                    64, 4.3, 120, False, False, False, False,
                    False, False, False, tYear=5)
    npt.assert_almost_equal(tmp, 0.015, decimal=3)
    tmp = qdiabetes(64, False, 27.34, 0.5, False, False,
                    64, 4.3, 120, False, False, False, False,
                    False, False, False, tYear=1)
    npt.assert_almost_equal(tmp, 0.003, decimal=3)
    tmp = qdiabetes(64, False, 27.34, 0.5, False, False,
                    64, 4.3, 120, False, False, False, False,
                    False, False, False, tYear=3)
    npt.assert_almost_equal(tmp, 0.008, decimal=3)
    tmp = qdiabetes(64, False, 27.34, 0.5, False, False,
                    64, 4.3, 120, False, False, False, False,
                    True, True, False, tYear=1)
    npt.assert_almost_equal(tmp, 0.014, decimal=3)
    tmp = qdiabetes(64, False, 27.34, 0.5, False, False,
                    64, 4.3, 120, False, False, False, True,
                    True, True, False, tYear=1)
    npt.assert_almost_equal(tmp, 0.016, decimal=3)
    tmp = qdiabetes(64, False, 27.34, 2, False, False,
                    64, 4.3, 120, False, False, False, True,
                    True, True, False, tYear=1)
    npt.assert_almost_equal(tmp, 0.0225, decimal=4)
    tmp = qdiabetes(64, False, 27.34, 8, False, False,
                    64, 4.3, 120, False, False, True, False,
                    True, True, False, tYear=1)
    npt.assert_almost_equal(tmp, 0.032, decimal=3)
    tmp = qdiabetes(64, False, 27.34, 8, False, True,
                    64, 4.3, 120, False, False, True, False,
                    True, True, False, tYear=1)
    npt.assert_almost_equal(tmp, 0.031, decimal=3)
    tmp = qdiabetes(64, True, 27.34, 0.5, False, False,
                    64, 4.3, 120, False, False, False, False,
                    True, False, True, tYear=1)
    npt.assert_almost_equal(tmp, 0.016, decimal=3)
    tmp = qdiabetes(64, True, 27.34, 0.5, False, False,
                    64, 4.3, 120, False, False, False, False,
                    True, False, True, tYear=2)
    npt.assert_almost_equal(tmp, 0.032, decimal=3)
    tmp = qdiabetes(64, True, 27.34, 0.5, False, False,
                    64, 4.3, 120, False, False, False, False,
                    True, False, True, tYear=5)
    npt.assert_almost_equal(tmp, 0.084, decimal=3)
    tmp = qdiabetes(64, True, 27.34, 4, False, False,
                    64, 4.3, 120, False, False, False, True,
                    True, False, True, tYear=1)
    npt.assert_almost_equal(tmp, 0.024, decimal=3)
    tmp = qdiabetes(64, True, 27.34, 0.5, True, False,
                    64, 4.3, 120, False, False, False, False,
                    True, False, True, tYear=1)
    npt.assert_almost_equal(tmp, 0.011, decimal=3)
    tmp = qdiabetes(64, True, 27.34, 0.5, False, True,
                    64, 4.3, 120, False, False, False, False,
                    True, False, True, tYear=1)
    npt.assert_almost_equal(tmp, 0.012, decimal=3)
    tmp = qdiabetes(64, True, 27.34, 5, False, True,
                    64, 4.3, 120, False, True, False, False,
                    True, False, True, tYear=1)
    npt.assert_almost_equal(tmp, 0.025, decimal=3)


def test_qdiabetes_json():
    model = QDiabetes(1)
    tmp = model.score({"index_age": 64,
                       "male": True,
                       "bmi": 27.34,
                       "diab_dur": 5,
                       "AC": False,
                       "EAsian": True,
                       "hba1c_mmol": 64,
                       "tchdl": 4.3,
                       "sbp": 120,
                       "heavy_smoke": False,
                       "moderate_smoke": True,
                       "light_smoke": False,
                       "prev_smoke": False,
                       "afib": True,
                       "cvd_hist": False,
                       "renal": True
    })
    npt.assert_almost_equal(tmp, 0.025, decimal=3)
