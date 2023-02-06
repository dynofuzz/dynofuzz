import os
from warnings import warn

# General
date = "20230112"
InformationRootDir = "data"

if not os.path.isdir(InformationRootDir):
    warn(f"{InformationRootDir} does not exist!")
parallel = 32

# Tree Generation
DUMMY_CONST = [1, 2]
EnableDiv = True
TreeMaxHeight = 5
TreeArgCount = 5

AggresiveEmpiricalPruning = True
EmpiricalPruning = True
EqualityPruning = True
ValuePruning = False