import numpy as np
import pandas as pd


def process(train_file, test_file, age_bins, fare_bins, out_dir):
    # Read data
    data = pd.read_csv(train_file)

    X_train = data.drop('Survived', 1)
    y_train = data['Survived'].to_frame()

    X_test = pd.read_csv(test_file)
    X_test_passenger_id = X_test['PassengerId'].to_frame()

    # Combine train and test set for easier processing
    train_rows_count = X_train.shape[0]
    combined_data = X_train.append(X_test, ignore_index=True)

    #
    # Column processing
    #

    # Name
    t = combined_data['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    titles = pd.get_dummies(t, prefix='Title')
    combined_data = pd.concat([combined_data, titles], axis=1)
    combined_data.drop('Name', 1, inplace=True)

    # Cabin
    d = combined_data['Cabin'].fillna('U').map(lambda c: "Deck_" + c.strip().upper()[0])
    decks = pd.get_dummies(d).drop('Deck_U', axis=1)
    combined_data = pd.concat([combined_data, decks], axis=1)
    combined_data.drop('Cabin', 1, inplace=True)

    # Ticket
    def p1(t):
        v = t.replace('.', '').upper().replace('A/5', 'A5').replace('A/4', 'A4').split(' ')
        if len(v) == 2:
            return v[0]
        else:
            return 'EMPTY'

    t1 = combined_data['Ticket'].map(p1)
    ticket_letters = pd.get_dummies(t1, prefix='Ticket').drop('Ticket_EMPTY', axis=1)
    combined_data = pd.concat([combined_data, ticket_letters], axis=1)

    # Ticket
    combined_data.drop('Ticket', 1, inplace=True)

    # Sex
    combined_data['Sex'].replace(to_replace='male', value=0, inplace=True)
    combined_data['Sex'].replace(to_replace='female', value=1, inplace=True)

    # Embarked
    combined_data['Embarked'].replace(to_replace='Q', value=1, inplace=True)
    combined_data['Embarked'].replace(to_replace='S', value=2, inplace=True)
    combined_data['Embarked'].replace(to_replace='C', value=3, inplace=True)
    combined_data['Embarked'].fillna(value=2, inplace=True)

    # Age
    combined_data['Age'].fillna(value=combined_data['Age'].mean(), inplace=True)
    combined_data['Age'] = (combined_data['Age'] / age_bins).apply(np.ceil)

    # Fare
    combined_data['Fare'].fillna(value=combined_data['Fare'].mean(), inplace=True)
    combined_data['Fare'] = (combined_data['Fare'] / fare_bins).apply(np.ceil)

    # Drop Passenger ID - it's not needed for training
    combined_data.drop('PassengerId', 1, inplace=True)

    #
    # End of column processing
    #

    # Split back to training and test data set
    X_train = combined_data[0:train_rows_count]
    X_test = combined_data[train_rows_count:]

    # Save
    X_train.to_csv("%s/x_train.csv" % out_dir, index=False)
    y_train.to_csv("%s/y_train.csv" % out_dir, index=False)
    X_test.to_csv("%s/x_test.csv" % out_dir, index=False)
    X_test_passenger_id.to_csv("%s/x_test_passenger_ids.csv" % out_dir)

