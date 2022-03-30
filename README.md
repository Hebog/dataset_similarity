# Dataset Similarity Ranking

This project was done for a Capita Selecta project at the 
Eindhoven University of Technology. This project can give a dataset similarity ranking based on all 
datasets in the [OpenML CC-18](https://www.openml.org/s/99) classification suite. Two ranking 
are used, one uses the Dataset2Vec meta-feature extractor as described in the paper "[Dataset2Vec: Learning Dataset Meta-Features](https://link.springer.com/article/10.1007/s10618-021-00737-9)."
and the other uses the [PyMFE](https://pymfe.readthedocs.io/en/latest/about.html) meta-features.

Next to producing two similarity rankings, these rankings can be evaluated using the meta-learning task of 
model selection.


## Usage
To produce a similarity ranking, run the rank_data_set_similarity.py file, with and OpenML dataset id as input.
For example:
```
python rank_dataset_similarity.py --input_dataset 14 
```

The meta-features can be re-extracted by:
```
python extract_meta_features.py 
```
This will output two csv files in the folder /extracted_MF

The performance of the two similarity rankings, based on the previously extracted meta-features,
can be evaluated by running:
```
python evaluate_similarity.py 
```
This outputs a csv file in the folder /similarity_evaluation. _Note: this has a long running time._
