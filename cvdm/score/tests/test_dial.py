import numpy.testing as npt

from cvdm.score import dial, Dial


def test_dial():
    tmp = dial(True, 55, 27, False,
               150, 5, 55, 70,
               False, False,
               5, True, False)
    npt.assert_almost_equal(tmp, 0.0248, decimal=4)
    

def test_dial_json():
    dial = Dial()
    tmp = dial.score({"index_age": 55,
                      "male": True, 
                      "bmi": 27,
                      "cur_smoke": False,
                      "sbp": 150,
                      "nonhdl_mmoll": 5,
                      "hba1c_mmoll": 55,
                      "egfr": 70,
                      "microalbum": False,
                      "macroalbum": False,
                      "diab_dur": 5,
                      "cvd_hist": True,
                      "insulin": False})
    npt.assert_almost_equal(tmp, 0.0248, decimal=4)
    
