# Data

Reference CFD data used for supervised boundary/data loss terms.

## Source
- Generated using **ANSYS Fluent** (2D incompressible NS, steady-state)
- Re range: 100 – 1000
- Grid: 200×100 structured mesh around cylinder

## Format
```
data/
  re_100.csv   # columns: x, y, u, v, p
  re_200.csv
  ...
  re_1000.csv
```

> Raw CFD data files are not included in this repo due to size.
> Contact author for access or reproduce using ANSYS Fluent with provided mesh parameters.
