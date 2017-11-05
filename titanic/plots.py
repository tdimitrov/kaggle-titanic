import matplotlib.pyplot as plt
import numpy as np

BAR_WIDTH = 0.35


def column(data, col_name):
    unq = np.sort(data[col_name].unique())

    survived = []
    died = []
    for c in unq:
        s = data[(data[col_name] == c) & (data['Survived'] == 1)]['PassengerId'].count()
        d = data[(data[col_name] == c) & (data['Survived'] == 0)]['PassengerId'].count()
        survived.append(s)
        died.append(d)

    bars_count = len(unq)

    fig, ax = plt.subplots()
    ind = np.arange(bars_count)
    p1 = ax.bar(ind, survived, color='blue', width=BAR_WIDTH)
    p2 = ax.bar(ind, died, color='red', width=BAR_WIDTH, bottom=survived)

    ax.set_title('Survivors by %s' % col_name)
    ax.legend((p1[0], p2[0]), ('Survived', 'Died'))
    ax.set_xticks(ind + BAR_WIDTH / 2)
    ax.set_xticklabels(unq)
    ax.set_ylabel('Number of people')


def binned_column(data, col, bin_size):
    bin_cnt = int(data[col].max() / bin_size)
    survived = data[data['Survived'] == 1][col].value_counts(sort=False, bins=bin_cnt)
    died = data[data['Survived'] == 0][col].value_counts(sort=False, bins=bin_cnt)

    labels = []
    prev = 0
    for curr in np.arange(bin_size, bin_cnt * bin_size + 1, bin_size):
        labels.append("%d-%d" % (prev + 1, curr))
        prev = curr

    bars_count = bin_cnt

    fig, ax = plt.subplots()
    ind = np.arange(bars_count)
    p1 = ax.bar(ind, survived, color='blue', width=BAR_WIDTH)
    p2 = ax.bar(ind, died, color='red', width=BAR_WIDTH, bottom=survived)

    ax.set_title('Survivors by %s, grouped in bins of %d' % (col, bin_size))
    ax.legend((p1[0], p2[0]), ('Survived', 'Died'))
    ax.set_xticks(ind + BAR_WIDTH / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Number of people')

