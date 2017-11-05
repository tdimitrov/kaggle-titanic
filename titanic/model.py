import pandas as pd
import numpy as np
import sklearn.svm as sksvm
import sklearn.model_selection as skms
import sklearn.preprocessing as sklpr
import matplotlib.pyplot as plt

pd.options.display.max_columns = 0


def svm(X_train, y_train):
    best_score = -1
    best_c = -1
    best_est = None

    print("Training a SVM estimator")

    for c in [3.3]:  # np.arange(3, 3.6, 0.01):
        m = sksvm.SVC(C=c)
        s = skms.cross_val_score(m, X_train, y_train).mean()

        if(best_score == -1) or (s > best_score):
            best_score = s
            best_c = c
            best_est = m

    print("Best score: %f, with c=%f" % (best_score, best_c))

    best_est.fit(X_train, y_train)
    return best_est


def make_prediction(estimator, X_test, X_test_passenger_ids, output_file):
    print("Generating predictions to %s" % output_file)
    p_survived = estimator.predict(X_test)
    prediction = pd.DataFrame({'Survived': p_survived}, index=X_test_passenger_ids)
    prediction.index.names = ['PassengerId']
    prediction.to_csv(output_file)


def plot_learning_curves(estimator, X_train, y_train):
    print("Plotting learning curves")
    cv = skms.ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    plt.figure()
    plt.title("Learning curves")

    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = skms.learning_curve(
        estimator, X_train, y_train, cv=cv, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

