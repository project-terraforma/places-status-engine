`schema_a.py` - Schema A to extract and engineer features from the three thousand examples in the overture places parquet. Originally just used the current features which yielded OK results for a baseline, tuned a couple parameters but overall did not accomplish much. Figured out that the signal exists in calculating the difference from Current Data to Base Data and using those `delta` feature columns to train the models, eliminating noise in the base and current features. 

`logistic_regression.py` 
- Trained on TF-IDF Vectorized columns for name primary and category primary
- Trained on OHE Categorical columns (region, country)
- Trained on Standard Scaled numeric columns on just delta columns
- Current version yielding ~85% percision for open ~67% percision for closed 
- ~75% recall for open and ~80% recall for closed. 

`random_forest.py`
- Trained on OHE Categorical columns (region, country)
- Trained on Standard Scaled numeric columns on just delta columns
- 88% percision on open, 64% percision on closed
- 70% recall on open, 84% recall on closed