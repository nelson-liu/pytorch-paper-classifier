# pytorch-paper-classifier

This is a bare-metal PyTorch implementation of the
[AllenNLP academic paper classifier example](https://github.com/allenai/allennlp-as-a-library-example),
which demonstrates how to build a model with AllenNLP as a dependency.

## Setting Up

The code in this project was developed with Python 3.6, so we recommend using that to run it.
However, it was written as much as possible with cross-version support in mind, but it's 
untested on other Python versions.

[Conda](https://conda.io/) will set up a virtual environment with the exact version of Python
used for development along with all the dependencies needed to run the code.

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Change your directory to your clone of this repo.

    ```
    cd pytorch-paper-classifier
    ```

3.  Create a Conda environment with Python 3.

    ```
    conda create -n paper_classifier python=3.6
    ```

4.  Now activate the Conda environment.
    You will need to activate the Conda environment in each terminal in which you 
    want to run code from this repo.

    ```
    source activate paper_classifier
    ```

5.  Install the required dependencies.

    ```
    pip install -r requirements.txt
    ```
    
6.  Install the SpaCy English model.

    ```
    python -m spacy download en
    ```

7. Visit [http://pytorch.org](http://pytorch.org) and install the relevant PyTorch 0.3.1 package (latest as of mid-March 2018).

You should now be able to test your installation with `pytest -v`.  Congratulations!

## Downloading the data

To download the papers data and a gzipped GloVe vectors file, run:

```
./download_data.sh
```

This will download training and validation data to a `./data/` subdirectory,
as well as a gzipped file of GloVe pretrained word vectors. The code looks
for the data in this directory by default.

## Basic Usage

To train a model with the default hyperparameters, simply run:

```
python run_model.py --save-dir ./saved_models/test
```

In this case, the model is saved to a directory at `./saved_models/test/`
(it will be created if it doesn't already exist).

To see a list of all the different hyperparameters you can tweak, run:

```
python run_model.py -h
```

### Visualizing Results

During and after training, you can visualize your results with `Tensorboard`
(you have to install this separately, and there are a variety of ways to do so).
To run Tensorboard while / after training the model above, run:

```
tensorboard --logdir saved_models/
```

and then navigate to http://localhost:6006/ (or http://hostname:6006/ , if running on a remote machine).
