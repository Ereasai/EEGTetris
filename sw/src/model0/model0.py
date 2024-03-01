import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import butter, lfilter
from joblib import dump
from custom_transformers import FilterBank

# Pipeline setup
pipeline = Pipeline([
    ('filterbank', FilterBank()),
    ('csp', CSP(n_components=4, transform_into='average_power')),
    ('lda', LinearDiscriminantAnalysis())
])

# Example usage with your data
if __name__ == '__main__':
    # Load your data here
    # For demonstration, I'll use placeholders for X and y
    # 100 samples, 8 channels, 250 time points.
    X, y = np.random.rand(100, 8, 250), np.random.randint(0, 2, 100)  # Example data shape and labels

    # Fit the pipeline to your data
    pipeline.fit(X, y)

    # Save the pipeline
    dump(pipeline, 'csp_lda_pipeline.joblib')

    print("Pipeline saved successfully.")
