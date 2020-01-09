# Automatic tag suggestions using a deep learning recommender system

This is the source code to my master's thesis.

Parts of the deep learning model is inspired by and adapted from the work of Xiangnan He https://github.com/hexiangnan/neural_collaborative_filtering.

## Setup

Install the required packages, noted in the environment.yml file. Note that this configuration of packages is made for gpu-computing.

Download dataset by following the instructions at https://multimediacommons.wordpress.com/yfcc100m-core-dataset/.

Run the preprocessing scripts in `src/preprocessing`, `preprocessing.py` first and then `generate_dev_test_data.py`.

The models are trained and tested by running the run-scripts in the respective `src/` folder, either `src/deep_learning/` for the
deep learning model or the `src/baseline/` folder for the baseline model.

The particular model configurations are specified in special yml run files.
See the `run_template.yml` files or yml files from past runs in the runs/ folders in either the `src/deep_learning/` or the `src/baseline/` folders for examples.
Several run_files can be queued by creating them and putting them in the folder `src/<model_name>/runs/several_runs/` and then running the `run_several.py` script.
In the case of the baseline model, an optimization of hyperparameters can be defined and run using the run_optimization.py file.
Several optimizations can be defined and queued using the `run_several_optimizations.py` script.

The deep learning model is tested separately using the `run_model_test.py` script.
