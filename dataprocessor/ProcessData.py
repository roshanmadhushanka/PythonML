import pandas as pd
import numpy as np
from featureeng import Math, Select, DataSetSpecific, Progress

def testData(moving_average=False, moving_median=False, standard_deviation=False, moving_entropy=False, entropy=False, probability_distribution=False):
    print "Testing frame process has started"
    print "---------------------------------"
    # Test data set preprocessor
    testing_frame = pd.read_csv("test.csv")
    ground_truth = pd.read_csv("rul.csv")

    # Obtain all column names
    all_column_names = list(testing_frame.columns)

    # Selected column names
    selected_column_names = all_column_names[5:]

    # Select seperation points to apply moving operations
    indices = Select.indices_seperate(feature_name="UnitNumber", data_frame=testing_frame)

    # Total work - progress
    total_work = len(selected_column_names)

    if moving_average:
        # Moving average window 5
        current_work = 0
        print "Applying Moving Average"

        for column_name in selected_column_names:
            current_work += 1
            column = testing_frame[column_name]
            slices = Select.slice(data_column=column, indices=indices)

            ma_header = "ma_5_" + column_name
            ma_calculated_array = np.array([])
            for slice in slices:
                ma_calculated_array = np.concatenate(
                    (ma_calculated_array, Math.moving_average(series=slice, window=5, default=True)), axis=0)
            testing_frame[ma_header] = pd.Series(ma_calculated_array, index=testing_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                   suffix="Complete")

    if moving_median:
        # Moving median window 5
        current_work = 0
        print "Applying Moving Median"

        for column_name in selected_column_names:
            current_work += 1
            column = testing_frame[column_name]
            slices = Select.slice(data_column=column, indices=indices)

            mm_header = "mm_5_" + column_name
            mm_calculated_array = np.array([])
            for slice in slices:
                mm_calculated_array = np.concatenate(
                    (mm_calculated_array, Math.moving_median(series=slice, window=5, default=True)), axis=0)
            testing_frame[mm_header] = pd.Series(mm_calculated_array, index=testing_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                   suffix="Complete")

    if standard_deviation:
        # Moving entropy
        current_work = 0
        print "Applying Standard Deviation"

        for column_name in selected_column_names:
            current_work += 1
            column = testing_frame[column_name]
            slices = Select.slice(data_column=column, indices=indices)

            sd_header = "sd_10_" + column_name
            sd_calculated_array = np.array([])
            for slice in slices:
                sd_calculated_array = np.concatenate(
                    (sd_calculated_array, Math.moving_standard_deviation(series=slice, window=10, default=True)), axis=0)
            testing_frame[sd_header] = pd.Series(sd_calculated_array, index=testing_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                   suffix="Complete")

    if moving_entropy:
        # Moving entropy
        current_work = 0
        print "Applying Moving Entropy"

        for column_name in selected_column_names:
            current_work += 1
            column = testing_frame[column_name]
            slices = Select.slice(data_column=column, indices=indices)

            me_header = "me_10_5_" + column_name
            me_calculated_array = np.array([])
            for slice in slices:
                #me_calculated_array = np.concatenate((me_calculated_array, Math.moving_entropy(series=slice, window=10, no_of_bins=5, default=True)), axis=0)
                me_calculated_array = np.concatenate((me_calculated_array, Math.moving_entropy(series=slice, no_of_bins=250)), axis=0)
            testing_frame[me_header] = pd.Series(me_calculated_array, index=testing_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                   suffix="Complete")

    if entropy:
        # Entropy
        current_work = 0
        print "Applying Entropy"

        for column_name in selected_column_names:
            current_work += 1
            column = testing_frame[column_name]
            e_header = "entropy_250_" + column_name
            testing_frame[e_header] = pd.Series(Math.entropy(series=column, no_of_bins=250), index=testing_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                   suffix="Complete")

    if probability_distribution:
        # Probability distribution
        current_work = 0
        print "Applying Probability Distribution"
        for column_name in selected_column_names:
            current_work += 1
            column = testing_frame[column_name]
            p_header = "prob_250_" + column_name
            testing_frame[p_header] = pd.Series(Math.probabilty_distribution(series=column, no_of_bins=250), index=testing_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                   suffix="Complete")


    filtered_frame = pd.DataFrame(columns=testing_frame.columns)

    # Add last index to the indices
    indices = np.insert(indices, len(indices), len(testing_frame['UnitNumber']) - 1, axis=0)

    # Select lines for test
    for index in indices:
        filtered_frame.loc[len(filtered_frame)] = testing_frame.loc[index]

    filtered_frame['RUL'] = pd.Series(ground_truth['RUL'], index=filtered_frame.index)

    print "Testing frame process is completed"
    filtered_frame.to_csv("Testing.csv", index=False)
    return filtered_frame

def trainData(moving_average=False, moving_median=False, standard_deviation=False, moving_entropy=False, entropy=False, probability_distribution=False):
    print "Training frame process has started"
    print "----------------------------------"
    # Data set preprocessor
    training_frame = pd.read_csv("train.csv")

    # Obtain all column names
    all_column_names = list(training_frame.columns)

    # Selected column names
    selected_column_names = all_column_names[5:-1]

    indices = Select.indices_seperate(feature_name="UnitNumber", data_frame=training_frame)

    # Total work - Progress
    total_work = len(selected_column_names)

    if moving_average:
        # Moving average window 5
        current_work = 0
        print "Applying Moving Average"

        for column_name in selected_column_names:
            current_work += 1
            column = training_frame[column_name]
            slices = Select.slice(data_column=column, indices=indices)

            ma_header = "ma_5_" + column_name
            ma_calculated_array = np.array([])
            for slice in slices:
                ma_calculated_array = np.concatenate(
                    (ma_calculated_array, Math.moving_average(series=slice, window=5, default=True)), axis=0)
            training_frame[ma_header] = pd.Series(ma_calculated_array, index=training_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress", suffix="Complete")

    if moving_median:
        # Moving median window 5
        current_work = 0
        print "Applying Moving Median"

        for column_name in selected_column_names:
            current_work += 1
            column = training_frame[column_name]
            slices = Select.slice(data_column=column, indices=indices)

            mm_header = "mm_5_" + column_name
            mm_calculated_array = np.array([])
            for slice in slices:
                mm_calculated_array = np.concatenate(
                    (mm_calculated_array, Math.moving_median(series=slice, window=5, default=True)), axis=0)
            training_frame[mm_header] = pd.Series(mm_calculated_array, index=training_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress", suffix="Complete")

    if standard_deviation:
        # Moving standard deviation 10
        current_work = 0
        print "Applying Standard Deviation"

        for column_name in selected_column_names:
            current_work += 1
            column = training_frame[column_name]
            slices = Select.slice(data_column=column, indices=indices)

            sd_header = "sd_10_" + column_name
            sd_calculated_array = np.array([])
            for slice in slices:
                sd_calculated_array = np.concatenate(
                    (sd_calculated_array, Math.moving_standard_deviation(series=slice, window=10, default=True)), axis=0)
            training_frame[sd_header] = pd.Series(sd_calculated_array, index=training_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress", suffix="Complete")

    if moving_entropy:
        # Moving entropy
        current_work = 0
        print "Applying Moving Entropy"

        for column_name in selected_column_names:
            current_work += 1
            column = training_frame[column_name]
            slices = Select.slice(data_column=column, indices=indices)

            me_header = "me_10_5_" + column_name
            me_calculated_array = np.array([])
            for slice in slices:
                me_calculated_array = np.concatenate((me_calculated_array, Math.moving_entropy(series=slice, window=10, no_of_bins=5, default=True)), axis=0)
            training_frame[me_header] = pd.Series(me_calculated_array, index=training_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                   suffix="Complete")

    if entropy:
        # Entropy
        current_work = 0
        print "Applying Entropy"

        for column_name in selected_column_names:
            current_work += 1
            column = training_frame[column_name]
            e_header = "entropy_250_" + column_name
            training_frame[e_header] = pd.Series(Math.entropy(series=column, no_of_bins=250), index=training_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress", suffix="Complete")

    if probability_distribution:
        # Probability distribution
        current_work = 0
        print "Applying Probability Distribution"
        for column_name in selected_column_names:
            current_work += 1
            column = training_frame[column_name]
            p_header = "prob_250_" + column_name
            training_frame[p_header] = pd.Series(Math.probabilty_distribution(series=column, no_of_bins=250), index=training_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                   suffix="Complete")

    # Add remaining useful life
    time_column = training_frame['Time']
    rul = DataSetSpecific.remaining_usefullifetime(indices=indices, time_series=time_column)
    training_frame['RUL'] = pd.Series(rul, index=training_frame.index)

    print "Training frame process is completed"
    training_frame.to_csv("Training.csv", index=False)
    return training_frame
