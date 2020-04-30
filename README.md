# cvdm

Cardiovascular Risk Scores for Patients with Diabetes

This Python package implements a variety of cardiovascular risk scores models.

## Installing

To install from the source:

```
$ git clone git@github.com:joyceho/cvdm.git
$ cd cvdm
$ python setup.py develop
```


## Example Code

Risk Prediction using Framingham Score (Simplified version with BMI)

```
from cvdm.score import frs_simple

risk = frs_simple(True, 35, 24.3, 122, False, True, False)
```

