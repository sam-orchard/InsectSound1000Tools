import os
import re
import shutil
from fetchfiles import fetchfiles

copy_or_move = 'copy'

input_dir = './InsectSound1000'
base_dir = './InsectSound1000_training'

# labels = ['Coccinella',
#           'Episyrphus',
#           'Bombus',
#           'Rhaphigaster',
#           'Bradysia',
#           'Aphidoletes',
#           'Halyomorpha',
#           'Nezara',
#           'Palomena',
#           'Trialeurodes',
#           'Myzus',
#           'Tuta']

labels = [
    'episyrphus',
]

# Directories for our training,
# validation and test splits
train_dir = os.path.join(base_dir, 'train')
# validation_dir = os.path.join(base_dir, 'validation')
validation_dir = os.path.join(base_dir, 'train') # Little hack to just create a ~20% test split with the rest in train
test_dir = os.path.join(base_dir, 'test')

date_regex = re.compile(r'^\d+[^-]') # Gets first string of numbers corresponding to date before first '-'

for label in labels:
    species_train_dir = os.path.join(train_dir, label)
    species_validation_dir = os.path.join(validation_dir, label)
    species_test_dir = os.path.join(test_dir, label)

    if not os.path.exists(species_train_dir):
        os.mkdir(species_train_dir)
    if not os.path.exists(species_test_dir):
        os.mkdir(species_test_dir)

    this_label_files = fetchfiles(input_dir, dir_keyword=label)
    print(str(len(this_label_files)) + ' files found for ' + label)

    # get all recording dates:
    dates = []
    for file in this_label_files:
        _, fname = os.path.split(file)
        match = date_regex.match(fname)
        if match is not None:
            date = match.group()
            if date not in dates:
                dates.append(date)

    # use list compression to build a dates dictionary for this label:
    dates_dict = {}
    for date in dates:
        dates_dict[date] = [file for file in this_label_files if date in file]

    # sort by length of list:
    dates_dict = dict(sorted(dates_dict.items(), key=lambda item: len(item[1]), reverse=True))
    # print sorted dict:
    for key, value in dates_dict.items():
        print(key, len(value))

    # now, let's split the files into train, validation and test sets:
    # make sure dates don't overlap between sets:
    train_files = []
    validation_files = []
    test_files = []

    if len(dates_dict) < 3:
        print(f'{label} - Not enough dates to split the data into train, validation and test sets.')
        break
    else:
        # biggest date goes to train, second biggest to validation, third biggest to test
        # rest gets added to train:
        # put the big files in first:
        train_files.extend(dates_dict.pop(next(iter(dates_dict))))
        validation_files.extend(dates_dict.pop(next(iter(dates_dict))))
        test_files.extend(dates_dict.pop(next(iter(dates_dict))))

        while len(dates_dict) > 0:

            # but make sure we have at least 1000 samples in the test set:
            while (len(test_files) / len(this_label_files)) < 0.15:
                if len(dates_dict) == 0:
                    break
                # add the smallest date to the test set:
                test_files.extend(dates_dict.pop(next(reversed(dates_dict))))

            # but make sure the validation set is not to small:
            while (len(validation_files) / len(this_label_files)) < 0.15:
                if len(dates_dict) == 0:
                    break
                # add the smallest left date to validation set:
                validation_files.extend(dates_dict.pop(next(reversed(dates_dict))))

            # add the rest to the train set:
            if len(dates_dict) > 0:
                train_files.extend(dates_dict.pop(next(reversed(dates_dict))))

    print('%s %s %s' % (str(len(train_files)), str(len(validation_files)), str(len(test_files))))
    print(str(round(len(train_files) / len(this_label_files) * 100, 2)) + '% '
          + str(round(len(validation_files) / len(this_label_files) * 100, 2)) + '% '
          + str(round(len(test_files) / len(this_label_files) * 100, 2)) + '%')
    print()

    # now, let's copy the files to the new directories:
    for files, dst_folder in zip([train_files, validation_files, test_files], [species_train_dir, species_validation_dir, species_test_dir]):
        for file in files:
            dontneedthis, fname = os.path.split(file)
            src = file
            dst = os.path.join(dst_folder, fname)
            if copy_or_move == 'move':
                shutil.move(src, dst)
            elif copy_or_move == 'copy':
                shutil.copy(src, dst)