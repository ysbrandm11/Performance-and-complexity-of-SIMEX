The complexity and performance of the SIMEX algorithm.

This code is made in addition to a thesis on the SIMEX algorithm. The goal of
the code is to give a clear understanding of the influence of the choice of
parameters on the complexity (time and space) and performance of SIMEX.
The code is written in Python and relies manely on numpy, matplotlib and sklearn
. The code has a working SIMEX algorithm, used on logistic regression.
The code is structured in a way that it is easy to use and understand.

The code is divided over four files:
-SIMEX.py: contains the simulation part of the SIMEX algorithm. Since for it to
work it needs a appropriate regression function for extrapolation.
- SIMEX_logistic_complexity.py: performs complexity analysis on the SIMEX
algorithm.
- SIMEX_logistic_performance.py: performs performance analysis on the SIMEX
algorithm.
- Process_data: Is used to process the data created by
SIMEX_logistic_performance.py.

IJsbrand Meeter
University of Amsterdam
13880624
ysbrandm@xs4all.nl
30 june 2024

