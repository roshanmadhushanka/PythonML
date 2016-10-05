import pandas as pd
import numpy as np
from featureeng import Math, Select, DataSetSpecific, Progress, ProbabilityOut
from file import FileHandler

_moving_average_window = 5
_moving_standard_deviation_window = 10
_moving_probability_window = 10

def testData(moving_average=False, moving_median=False, standard_deviation=False, moving_entropy=False, entropy=False, probability_distribution=False, moving_probability=False, probability_from_file=False, moving_k_closest_average=False, moving_threshold_average=False, moving_median_centered_average=False, moving_weighted_average=False,rul=True):
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
                me_calculated_array = np.concatenate((me_calculated_array, Math.moving_entropy(series=slice, window=10, no_of_bins=5, default=True)), axis=0)
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
            p_header = "prob_" + column_name
            testing_frame[p_header] = pd.Series(Math.probabilty_distribution(series=column, no_of_bins=250), index=testing_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                   suffix="Complete")
        ProbabilityOut.saveToFile()

    if moving_probability:
        # Moving probability distribution
        current_work = 0
        print "Applying Moving probability"
        for column_name in selected_column_names:
            current_work += 1
            column = testing_frame[column_name]
            p_header = "prob_" + column_name
            testing_frame[p_header] = pd.Series(Math.moving_probability(series=column, window=10, no_of_bins=4, default=True),
                                                index=testing_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                   suffix="Complete")

    if probability_from_file:
        # Load probabilities from file
        file_name = 'json.txt'
        current_work = 0
        print "Applying Probability From File"
        for column_name in selected_column_names:
            current_work += 1
            column = testing_frame[column_name]
            p_header = "prob_" + column_name
            testing_frame[p_header] = pd.Series(from_file(column, column_name),
                index=testing_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                   suffix="Complete")

    if moving_k_closest_average:
        # Moving k closest average
        current_work = 0
        print "Applying K Closest Average"
        for column_name in selected_column_names:
            current_work += 1
            column = testing_frame[column_name]
            p_header = "k_closest_" + column_name
            testing_frame[p_header] = pd.Series(Math.moving_k_closest_average(series=column, window=5, kclosest=3, default=True),
                index=testing_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                   suffix="Complete")

    if moving_threshold_average:
        # Moving threshold average
        current_work = 0
        print "Applying Threshold Average"
        for column_name in selected_column_names:
            current_work += 1
            column = testing_frame[column_name]
            p_header = "threshold_" + column_name
            testing_frame[p_header] = pd.Series(
                Math.moving_threshold_average(series=column, window=5, threshold=-1, default=True),
                index=testing_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                   suffix="Complete")

    if moving_median_centered_average:
        # Moving median centered average
        current_work = 0
        print "Applying Median Centered Average"
        for column_name in selected_column_names:
            current_work += 1
            column = testing_frame[column_name]
            p_header = "threshold_" + column_name
            testing_frame[p_header] = pd.Series(
                Math.moving_median_centered_average(series=column, window=5, boundary=1, default=True),
                index=testing_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                   suffix="Complete")

    if moving_weighted_average:
        # Moving weighted average
        current_work = 0
        print "Applying Weighted Average"
        for column_name in selected_column_names:
            current_work += 1
            column = testing_frame[column_name]
            p_header = "threshold_" + column_name
            testing_frame[p_header] = pd.Series(
                Math.moving_weighted_average(series=column, window=5, weights=[5, 4, 3, 2, 1], default=True),
                index=testing_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                   suffix="Complete")




    filtered_frame = pd.DataFrame(columns=testing_frame.columns)

    # Add last index to the indices
    indices = np.insert(indices, len(indices), len(testing_frame['UnitNumber']) - 1, axis=0)

    # Select lines for test
    for index in indices:
        filtered_frame.loc[len(filtered_frame)] = testing_frame.loc[index]

    if rul:
        filtered_frame['RUL'] = pd.Series(ground_truth['RUL'], index=filtered_frame.index)
        print "Applying RUL"

    print "Testing frame process is completed\n"
    filtered_frame.to_csv("Testing.csv", index=False)
    return filtered_frame

def trainData(moving_average=False, moving_median=False, standard_deviation=False, moving_entropy=False, entropy=False, probability_distribution=False, moving_probability=False, moving_k_closest_average=False, moving_threshold_average=False, moving_median_centered_average=False, moving_weighted_average=False):
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
            p_header = "prob_" + column_name
            training_frame[p_header] = pd.Series(Math.probabilty_distribution(series=column, no_of_bins=250), index=training_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                   suffix="Complete")

    if moving_probability:
        # Moving probability distribution
        current_work = 0
        print "Applying Moving probability"
        for column_name in selected_column_names:
            current_work += 1
            column = training_frame[column_name]
            p_header = "prob_" + column_name
            training_frame[p_header] = pd.Series(Math.moving_probability(series=column, window=10, no_of_bins=4, default=True),
                                                 index=training_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                       suffix="Complete")

    if moving_k_closest_average:
        # Moving k closest average
        current_work = 0
        print "Applying K Closest Average"
        for column_name in selected_column_names:
            current_work += 1
            column = training_frame[column_name]
            p_header = "k_closest_" + column_name
            training_frame[p_header] = pd.Series(
                Math.moving_k_closest_average(series=column, window=5, kclosest=3, default=True),
                index=training_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                       suffix="Complete")

    if moving_threshold_average:
        # Moving threshold average
        current_work = 0
        print "Applying Threshold Average"
        for column_name in selected_column_names:
            current_work += 1
            column = training_frame[column_name]
            p_header = "threshold_" + column_name
            training_frame[p_header] = pd.Series(
                Math.moving_threshold_average(series=column, window=5, threshold=-1, default=True),
                index=training_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                       suffix="Complete")

    if moving_median_centered_average:
        # Moving median centered average
        current_work = 0
        print "Applying Median Centered Average"
        for column_name in selected_column_names:
            current_work += 1
            column = training_frame[column_name]
            p_header = "threshold_" + column_name
            training_frame[p_header] = pd.Series(
                Math.moving_median_centered_average(series=column, window=5, boundary=1, default=True),
                index=training_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                       suffix="Complete")

    if moving_weighted_average:
        # Moving weighted average
        current_work = 0
        print "Applying Weighted Average"
        for column_name in selected_column_names:
            current_work += 1
            column = training_frame[column_name]
            p_header = "threshold_" + column_name
            training_frame[p_header] = pd.Series(
                Math.moving_weighted_average(series=column, window=5, weights=[5, 4, 3, 2, 1], default=True),
                    index=training_frame.index)
            Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
                                       suffix="Complete")

    # Add remaining useful life
    time_column = training_frame['Time']
    rul = DataSetSpecific.remaining_usefullifetime(indices=indices, time_series=time_column)
    training_frame['RUL'] = pd.Series(rul, index=training_frame.index)

    print "Training frame process is completed\n"
    training_frame.to_csv("Training.csv", index=False)
    return training_frame

def from_file(series, column_name, no_of_bins=250):
    file_name = "json.txt"
    data = FileHandler.read_json(file_name)

    rang = data[column_name]['rang'].split(",")
    x = map(float, rang)

    prob = data[column_name]['prob'].split(",")
    p = map(float, prob)

    bin_size = float(data[column_name]['sbin'])

    '''
            Calculate the probability of data for whole data set

            :param series: Input number series
            :param no_of_bins: Number of discrete levels
            :return: calculated result in numpy array
        '''
    series = list(series)

    # Calculate bin size
    min_value = x[0]

    '''
     Bin size becomes zero when the values in the series are not changing
     That means probability of occuring that value is 1 which means entropy is zero
    '''

    if bin_size == 0.0:
        # if value is not changing probability is one
        return np.ones(shape=len(series))

    ret = []
    for num in series:
        bin = int((num - min_value) / bin_size)
        if 0 <= bin < no_of_bins:
            ret.append(p[bin])
        elif bin == no_of_bins:
            ret.append(p[no_of_bins - 1])
        else:
            ret.append(0.0)
    return np.array(ret)
