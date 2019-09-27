Example files
=============
Each of these subfolders contain an example of how `config.ini` and
`true_model.py` might look for different surrogate modeling cases.
For instance, `rosenbrock` and `ripples` show two examples with 2D
input and 1D output, while `arrays` shows 2D input and output, and
`chebyshevs` demonstrates 1D input with 3D output. Each example can
be executed by opening a terminal in the relevant subfolder, and
running the following command from the terminal:
    python ../../../src/main.py
The output is written to the files `samples.csv` (raw data) and
`regression.csv` (model coefficients) in the same folder. The
resulting surrogate models are also written to the terminal.
