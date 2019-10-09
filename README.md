# LendingAtlas
A platform for assessing risk in micro-lending. Please find the slides associated with this project [here](https://drive.google.com/open?id=1ejxnE_gvRhf6n6RGqNMrEHwo2j7aHA5CkmFvz4YNKso).

## Motivation
Nearly 70% percent of the world population has no credit history and is ineligible for loans from almost every institution. Yet many companies and individuals would be willing to extend loans if there was a way to assess the risk. Improved understanding implies not just the generation of a somewhat-opaque score but explaining the factors that contribute to reducing the risk of default. Such a comprehensive picture would facilitate significant increase in lending and benefit both the lenders and borrowers.

When employing data to assess creditworthiness, the standard approach is to tabulate the available categorical information  &mdash; geographic location, gender, sector etc.  &mdash; and run logistic regression or random forest to produce a probability of default, i.e. a credit score. The resulting credit score, however accurate it may be, is not sufficient enable companies to expand their activities and individuals to confidently extend micro-loans. An alternative approach should provide insights that go beyond a single numerical value.

## Approach

Embeddings can be used to convert categorical data, which are typically represented using orthogonal unit-vectors in a high-dimensional space (one-hot vectors), to real-valued vectors in just a few dimensions. The resulting low-dimensional real-valued vectors may then be used (1) as inputs to a Multi Layer Perceptron or some other classification algorithm, or (2) to explore relationships in the data. The weight matrices that can be used to convert the categorical data to real-valued vectors are called "Embeddings" and are calculated by training a neural network on a supervised task.

## Data
Data originate from <a href="https://www.kiva.org"> kiva.org </a>, which provides micro-loan <a href="https://build.kiva.org/docs/data/snapshots"> data snapshots </a> at its website. However, kiva snapshots no longer include repayment status, only funded/expired information, and this had to be scraped and merged into the kiva dataset. A processed dataset, where loans are either fully paid or defaulted upon, is included under `/data/processed/processed_kiva_data.csv.zip`. The original kiva files and the preprocessed versions are omitted. Functions that can be used to generate the processed files starting from the "data snapshots" are available under `/src/utilities.py`.

## Requisites
The code was developed on Python 3.7 and requires the following libraries:
- scikit-learn
- keras
- argparse
- numpy 

After cloning the repository you can recreate the environment and install the package dependencies using:
```shell
$ conda env create -f build/LendingAtlas_env.yml
```

## Running the code

### Command line interface
The code can be run from the command line by calling:
```shell
$ python main.py
```
Two **output folders** will be generated:
- `output_figs`: Output figures `.png` for this dataset.
- `output_embeddings`: Embeddings `.csv` for the categorical variables.

The following **flags** are available:

- `--data`: The file path to the processed data.

- `--solver='All'` enables one to select between (1) 'Logistic Regression', (2) 'Random Forest', (3) 'Embeddings', (4) 'All'

- `--shallow_net 32 8`: A shallow network is used to train the embeddings and the list of positive integers represents the nodes per hidden layer.

- `--deep_net 64 64 64 8`: A neural network that uses the embeddings evaluated by the shallow network. By default this keyword is not present. If it is present it should be followed by the number of nodes per hidden layer.

- `--epochs 50 50`: First network (that evaluates the embeddings) is trained for 50 epochs (first value) and second network (that uses the pretrained embeddings is trained for another 50 epochs (second value). 

- `--batch_size 500 500`: Batch sizes for the first and second networks respectively.

- `--sample="undersample"`: By default if the classes are imbalanced the code undersamples the majority class to achieve 1 to 1 ratio. Alternatively one can set "oversample" or "None" to either oversample the mionority class (to 1 to 1 ratio) or to continue using the imbalanced dataset. 

- `--explore_data`: If this flag is present then a few exploratory plots are generated from the data. 

### Example
Run the code with:
```shell
$ python main.py --epochs 50 50 --batch_size 500 500 --shallow_net 32 8 --deep_net 64 64 64 8
```
The code will report precision, recall, and f1-score for all 4 models. More detailed metrics are reported in the `output_figs` directory where the confusion matrices for all 4 models along with the ROC curve and the Precision-Recall curve are plotted.

In `output_figs` the code will also show the T-distributed Stochastic Network Embedding (t-SNE) for each of the default embeddings. The color coding corresponds to the probability of default. The "Partner ID" is particularly prone to clustering between partners with high probability of default and those with low. The "Town" feature is also shown for specific countries and plotted. Below, towns in Guatemala are shown on a background of other towns (towns with more than 200 loans) in the dataset.

<p float="left">
  <img src="https://github.com/michail-tzoufras/LendingAtlas/blob/master/example_output/Partner%20ID_embedding_plot.png" width="400" />
  <img src="https://github.com/michail-tzoufras/LendingAtlas/blob/master/example_output/Guatemala_embedding_plot.png" width="400" /> 
</p>

Finally, the cosine similarity is shown between the "Lebanese LBP" of the "Country Currency" embedding and the most similar (as well as most dissimilar) values.

<p align="center">
  <img src="https://github.com/michail-tzoufras/LendingAtlas/blob/master/example_output/LebanesePound.png" 
       width="500" />
</p>
