# ML-EPFL
Solutions to Machine Learning Projects at EPFL\
Link to the [dataset and submission platform](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs).\
Register for this using your **EPFL email address** so that we can create a team for our submissions.

# Project Structure
- `src` directory contains all files related to the source code.
- `dataset` directory contains the dataset.
- `test` directory contains the code for testing / verifying correctness or accuracy.
- `docs` directory contains everything related to writing documentation for our work.
- `setup.sh` is the script to use for installing the dependencies into the machine which will run this project. (Supports installing dependencies for linux environments only).

# File Descriptions
**dataset/train.csv** - Training set of 250000 events. The file starts with the `ID` column, then the label column (the y you have to predict), and finally 30 feature columns.\
**dataset/test.csv** - The test set of around 568238 events - Everything as above, except the label is missing.\
**dataset/sample-submission.csv** - a sample submission file in the correct format. The sample submission always predicts -1, that is `background`.

For detailed information on the semantics of the features, labels, and weights, see the technical documentation from the LAL website on the task. Note that here for the EPFL course, we use a simpler evaluation metric instead (classification error).

#### Some details to get started:
- All variables are floating point, except PRI_jet_num which is integer
- Variables prefixed with `PRI` (for PRImitives) are “raw” quantities about the bunch collision as measured by the detector.
- Variables prefixed with `DER` (for DERived) are quantities computed from the primitive features, which were selected by the physicists of `ATLAS`.
- It can happen that for some entries some variables are meaningless or cannot be computed; in this case, their value is −999.0, which is outside the normal range of all variables.

# Rules
Each participant is allowed to make **5 submissions per day**. If you particpate as a team, the whole team gets 5 submissions, not 15 as the rules page states. Failed submissions (e.g. wrong submission file format) do not count.

For further information, please checkout the project description document at `docs/problem_statement.pdf`

# Authors
To-do: Add names of group members here.