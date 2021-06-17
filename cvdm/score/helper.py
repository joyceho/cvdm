
def clean_diab_dur(diab_dur, min_val=1):
    return max(min_val, diab_dur)


def clean_hba1c(hba1c, meas="per"):
    if meas == "per":
        return max(3, hba1c)
    else:
        return max(9, hba1c)

    
def clean_acr(acr):
    return max(1, acr)


def clean_pp(pp):
    return max(0, pp)


def clean_bp(bp):
    return max(10, bp)


def clean_height(height, meas="m"):
    if meas == "m":
        return max(0.2, height)
    else:
        # return it in inches
        return max(7.87, height)


def clean_egfr(egfr):
    return max(1, egfr)


def clean_nonhdl(nonhdl, meas="mgdl"):
    return max(0, nonhdl)


def clean_chol(chol, meas="mgdl"):
    return max(0, chol)


def clean_ldl(ldl, meas="mgdl"):
    return max(0, ldl)


def clean_hdl(hdl, meas="mgdl"):
    min_hdl_mmol = 0.01
    if meas == "mmol":
        return max(min_hdl_mmol, hdl)
    else:
        return max(min_hdl_mmol*38.67, hdl)


def clean_tot_chol(chol_tot, meas="mgdl"):
    return max(0, chol_tot)


def clean_tchdl(tchdl):
    return max(1, tchdl)


def clean_bmi(bmi):
    return max(10, bmi)


def clean_age(age):
    return max(18, age)


def clean_hb(hb):
    return max(0, hb)

