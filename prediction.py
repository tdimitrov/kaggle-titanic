import pandas as pd
import titanic as tt

X_train = pd.read_csv('data/processed/x_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
X_test = pd.read_csv('data/processed/x_test.csv')
X_test_passenger_ids = pd.read_csv('data/processed/x_test_passenger_ids.csv')

m = tt.model.svm(X_train, y_train['Survived'])

tt.model.make_prediction(m, X_test, X_test_passenger_ids['PassengerId'], 'data/prediction.csv')

tt.model.plot_learning_curves(m, X_train, y_train['Survived'])
