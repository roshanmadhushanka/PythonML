from featureeng import Math, Select, DataSetSpecific, Progress
import pandas as pd
import numpy as np

def process():
    print "Training frame process has started"
    print "----------------------------------"
    # Data set preprocessor
    training_frame = pd.read_csv("train.csv")

    # Obtain all column names
    all_column_names = list(training_frame.columns)

    # Selected column names
    selected_column_names = all_column_names[5:-1]

    indices = Select.indices_seperate(feature_name="UnitNumber", data_frame=training_frame)

    # Progress
    current_work = 0
    total_work = len(selected_column_names)


    # Moving average window 5
    print "Applying Moving Average"
    for column_name in selected_column_names:
        current_work += 1
        column = training_frame[column_name]
        slices = Select.slice(data_column=column, indices=indices)

        ma_header = column_name + "_ma_5"
        ma_calculated_array = np.array([])
        for slice in slices:
            ma_calculated_array = np.concatenate(
                (ma_calculated_array, Math.moving_average(series=slice, window=5, default=True)), axis=0)
        training_frame[ma_header] = pd.Series(ma_calculated_array, index=training_frame.index)
        Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress", suffix="Complete")

    current_work = 0
    # Moving median window 5
    print "Applying Moving Median"
    for column_name in selected_column_names:
        current_work += 1
        column = training_frame[column_name]
        slices = Select.slice(data_column=column, indices=indices)

        mm_header = column_name + "_mm_5"
        mm_calculated_array = np.array([])
        for slice in slices:
            mm_calculated_array = np.concatenate(
                (mm_calculated_array, Math.moving_median(series=slice, window=5, default=True)), axis=0)
        training_frame[mm_header] = pd.Series(mm_calculated_array, index=training_frame.index)
        Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress", suffix="Complete")

    current_work = 0
    # Moving standard deviation 10
    print "Applying Standard Deviation"
    for column_name in selected_column_names:
        current_work += 1
        column = training_frame[column_name]
        slices = Select.slice(data_column=column, indices=indices)

        sd_header = column_name + "_sd_10"
        sd_calculated_array = np.array([])
        for slice in slices:
            sd_calculated_array = np.concatenate(
                (sd_calculated_array, Math.moving_standard_deviation(series=slice, window=10, default=True)), axis=0)
        training_frame[sd_header] = pd.Series(sd_calculated_array, index=training_frame.index)
        Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress", suffix="Complete")

    # # Moving entropy
    # for column_name in selected_column_names:
    #     column = training_frame[column_name]
    #     slices = Select.slice(data_column=column, indices=indices)
    #
    #     me_header = column_name + "_me_10_5"
    #     me_calculated_array = np.array([])
    #     for slice in slices:
    #         me_calculated_array = np.concatenate((me_calculated_array, Math.moving_entropy(series=slice, window=10, no_of_bins=5, default=True)), axis=0)
    #     training_frame[me_header] = pd.Series(me_calculated_array, index=training_frame.index)

    current_work = 0
    # Entropy
    print "Applying Entropy"
    for column_name in selected_column_names:
        current_work += 1
        column = training_frame[column_name]
        e_header = column_name + "_entropy_250"
        training_frame[e_header] = pd.Series(Math.entropy(series=column, no_of_bins=250), index=training_frame.index)
        Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress", suffix="Complete")


    # current_work = 0
    # # Probability distribution
    # print "Applying Probability Distribution"
    # for column_name in selected_column_names:
    #     current_work += 1
    #     column = training_frame[column_name]
    #     p_header = column_name + "_prob_250"
    #     training_frame[p_header] = pd.Series(Math.probabilty_distribution(series=column, no_of_bins=250), index=training_frame.index)
    #     Progress.printProgress(iteration=current_work, total=total_work, decimals=1, prefix="Progress",
    #                            suffix="Complete")

    # Add remaining useful life
    time_column = training_frame['Time']
    rul = DataSetSpecific.remaining_usefullifetime(indices=indices, time_series=time_column)
    training_frame['RUL'] = pd.Series(rul, index=training_frame.index)

    print "Training frame process is completed"
    training_frame.to_csv("Training.csv", index=False)
    return training_frame






