# iris
Imperial College Storm Model

This is IRIS (v0.1), the Imperial College Storm Model. For a detailed description and analysis, see [Sparks & Toumi (2024)](https://www.nature.com/articles/s41597-024-03250-y). The 10,000 year global synthetic dataset described there is available [here](https://doi.org/10.6084/m9.figshare.c.6724251.v1).

The code (run/) and data (rundata/) needed to run IRIS are all here. The script

    run/run_iris.py 

will run IRIS using python multiprocessing and is set up to produce the sample 100 year global output which is included in numpy format here:

    out/

and in text format here:

    out_txt/
