# Music Data Analysis

Exploratory analysis of song features, clustering and popularity using Python.

This project looks at relationships between audio features, groups tracks with similar
profiles and compares three simple classifiers on artists held out of training.

**[Read the executed notebook](music_analysis.ipynb)** for the complete analysis and its outputs.

![Release-decade coverage, popularity scores and track-duration distributions](assets/sample_overview.png)

## Data and scope

The CSV contains **795 records and 16 columns**, with release years from
1927 to 2023. It includes artist and track names, a popularity score and
audio descriptors such as danceability, energy, loudness, acousticness, valence and tempo.

After removing 3 exact duplicate rows, **792 records**
remain. The original CSV is unchanged. One repeated artist/title pair has conflicting release
years and is kept for inspection. Some titles have apparent encoding errors.

About 76% of the analysed records date from 2000 onwards; the 1930s have
just one record. The analysis describes this sample rather than the history of music.

Source: [Music Genre Classification on Kaggle](https://www.kaggle.com/datasets/purumalgi/music-genre-classification),
uploaded by Purushottam Malgi from a MachineHack exercise. This is an adapted extract: it omits
`instrumentalness` and genre `Class`, and adds `release_year`. The preparation steps and origin
of the years were not preserved, so decade plots should be treated cautiously. Kaggle lists
the licence as **Other (specified in description)**; a standard open-data licence is not assumed.

## Approach

- Check data quality, inspect distributions and correlations, and flag IQR outliers.
- Compare K-Means with k=2 through k=10 on four standardised audio features. Use silhouette
  scores to select k and PCA to plot the groups.
- Compare K-Means and Ward clustering on a fixed 100-record sample using Adjusted Rand Index.
- Compare Decision Tree, Random Forest and Gaussian Naive Bayes with a majority-class baseline.
  Use five-fold group cross-validation for selection, then a separate test set for evaluation.

## Findings

### Song clusters

The highest silhouette score among the tested values selects **k=3**, with a score
of **0.249**. The PCA plot retains **69.1%** of the variance.
The groups describe audio profiles rather than verified genres.

![Selected song clusters projected onto two principal components](assets/song_clusters.png)

### Popularity classification

Class 1 means a recorded popularity score **strictly above 50**. This fixed
threshold defines the exercise; the model is not forecasting future hits.

The split contains **641 training records from 588 artist
groups** and **151 test records from 147 other artist groups**.
Grouping uses normalised credited-name strings; aliases and collaborations may still overlap.

Training-only cross-validation results:

| Model | Mean balanced accuracy | Fold standard deviation |
| --- | --- | --- |
| Gaussian Naive Bayes | 58.27% | 2.69% |
| Random Forest | 53.73% | 3.98% |
| Decision Tree | 52.73% | 5.39% |
| Majority baseline | 50.00% | 0.00% |

Cross-validation selects **Gaussian Naive Bayes**. Fold standard deviation is not a confidence interval.
Only that trained model and the baseline are then evaluated on the test set:

| Model | Accuracy | Balanced accuracy | F1, class 1 | ROC AUC |
| --- | --- | --- | --- | --- |
| Majority baseline | 68.21% | 50.00% | 0.000 | 0.500 |
| Gaussian Naive Bayes | 60.26% | 55.30% | 0.400 | 0.631 |

Naive Bayes has higher balanced accuracy but lower overall accuracy than the baseline,
which always predicts the more frequent class. Balanced accuracy averages recall across
both classes. The improvement is modest and needs testing on more data.

## Reproduce the analysis

Tested with **Python 3.12.14**. The CSV is included and no API keys are needed.

```bash
git clone https://github.com/nachopalomares/music-analysis.git
cd music-analysis
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab music_analysis.ipynb
```

On Windows, activate the environment with `.venv\Scripts\activate`. Keep the working
directory at the repository root and run all cells from a fresh kernel. To execute without
the Jupyter interface:

```bash
python -m nbconvert --to notebook --execute --inplace music_analysis.ipynb
```

The notebook saves charts to [`assets/`](assets/) and tables to [`results/`](results/).
The README's figures come from [`summary.json`](results/summary.json). All 13 code cells
were run from a fresh kernel, including checks for artist overlap and valid metrics.

## Limitations and next step

This is a small, previously explored sample. Its preparation is undocumented and one
release-year conflict remains. Artist grouping relies on names, so aliases and collaborations
can still overlap. Cluster stability and results on other catalogues, future periods or
different thresholds have not been tested. Documenting the extract and obtaining a larger
sample would be the next steps.
