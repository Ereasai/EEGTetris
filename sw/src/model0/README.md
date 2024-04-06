# Loading and Using the CSP-LDA Pipeline

## Requirements

Before loading the pipeline, ensure that you have installed all necessary Python packages. You can install them using the provided `requirements.txt` file by running:


## Loading the Pipeline

To load the pipeline, use the following Python code:

```python
from joblib import load

pipeline = load('csp_lda_pipeline.joblib')
```