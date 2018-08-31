# Kaggle Spooky Author Identification
My work on the [Kaggle Spooky Author Identification](https://www.kaggle.com/c/spooky-author-identification) dataset.

## Getting Started

### Project Requirements
1. Download Anaconda or Miniconda: https://conda.io/docs/user-guide/install/download.html.
1. The project was executed on Ubuntu 16.04 LTS powered by a GeForce GTX 1080 Ti to run random search on the neural network models. If you don't have TensorFlow installed on your system, please refer to their installation docs: https://www.tensorflow.org/install/.
1. View the requirements for this project in the `requirements.txt` file; note that you may need different package versions based on your operating system.

### Installation Steps
1. After you have Anaconda or Miniconda installed, run `conda create --name authorid python=3.6.5`.
1. Run `source activate authorid`.
1. Run `pip install -r requirements.txt`.
1. Run `KERAS_BACKEND=tensorflow python -c "from keras import backend"`.
1. Run `python -m ipykernel install --user --name authorid --display-name "authorid"`.
1. If you don't already have the required input files, perform the steps in **Download Input Files** first.
1. Run `jupyter notebook` and open one of the notebook files in the `code/` folder.
1. Select `Kernel > Change kernel > authorid` to change the kernel.

### Download Input Files
1. Create a Kaggle account and download the data from https://www.kaggle.com/c/spooky-author-identification/data (you might need to accept competition rules before you are allowed to retrieve the data). Download the 3 files `train.zip`, `test.zip`, and `sample_submission.zip`, unzip them, and add the resulting CSV files to the `input/` folder.
1. Download the GloVe `glove.840B.300d.zip` embeddings from https://nlp.stanford.edu/projects/glove/, unzip them, and add them to the `input/embeddings/` folder.
1. Download the fastText `crawl-300d-2M.vec.zip` embeddings from https://fasttext.cc/docs/en/english-vectors.html, unzip them, and add them to the `input/embeddings/` folder.

## Usage

### Creating a New Notebook for a Model
1. Open an existing model notebook in the `code/` folder, select `File > Make a Copy...` from the Jupyter menu, and change the name to the new model.
1. Change the title in the notebook to the new model.
1. Change the `MODEL_NAME` variable in cell 5 to the new model.
1. Change the `EMBEDDINGS_FILE_PATH` variable in cell 5 if necessary.
1. Change cell 12 ("Import model-dependent files") if necessary.

### Testing a Set Model without Random Search
1. Set the number of iterations `num_random_search_iter` to 1.
1. Comment out the line `random_model_params = get_random_model_params()` and replace it with `random_model_params = {}` (see the `models/__init__.py` file for details on what the parameters should be set to).
