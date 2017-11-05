# Solution for Kaggle's Titanic: Machine Learning from Disaster training competition.

## Intended usage

0. Optional step: Various plots of the input data
```bash
python ./plot_training_data.py
```
1. Process the input data. The transformed data is saved in data/processed
```bash
python ./process_data.py
```
2. Make a prediction - saved in data/prediction.csv
```bash
python prediction.py
```
3. To cleanup remove the following files/dirs:
* data/processed
* data/prediction.csv
