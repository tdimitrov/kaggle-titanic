import titanic as tt
import os
import sys

train_file='./data/input/train.csv'
test_file='./data/input/test.csv'
age_bins = 10
fare_bins = 25
out_dir = './data/processed/'

if os.path.exists(out_dir) == False:
    try:
        os.mkdir(out_dir)
        print("Created dir %s" % out_dir)
    except FileExistsError as e:
        print("%s exists, but it's not a directory. Exiting." % out_dir)
        sys.exit(1)

tt.data_processor.process(train_file, test_file, age_bins, fare_bins, out_dir)
