# Random Forest Simulator
# h2o testing
import h2o
from h2o.estimators import H2ORandomForestEstimator
from dataprocessor import ProcessData
from simulation import RandomForestResult
from featureeng import Progress

def randomforest_simulator():
    print "Random Forest simulator"
    print "-----------------------"

    tree_lim = [5, 51]
    depth_lim = [5, 21]
    print "No of trees : " + str(tree_lim[0]) + " to " + str((tree_lim[1]-1))
    print "Depth       : " + str(depth_lim[0]) + " to " + str((depth_lim[1]-1))

    tree_arr = range(tree_lim[0], tree_lim[1])
    depth_arr = range(depth_lim[0], depth_lim[1])

    job_no = 0
    total_work = len(tree_arr) * len(depth_arr)

    # Initialize server
    h2o.init()

    # define response column
    response_column = u'RUL'

    # load pre-processed data frames
    training_frame = ProcessData.trainData(moving_average=True, standard_deviation=True, moving_entropy=True)
    testing_frame = ProcessData.testData(moving_average=True, standard_deviation=True, moving_entropy=True)

    # create h2o frames
    train = h2o.H2OFrame(training_frame)
    test = h2o.H2OFrame(testing_frame)
    train.set_names(list(training_frame.columns))
    test.set_names(list(testing_frame.columns))

    # Feature selection
    training_columns = list(training_frame.columns)
    training_columns.remove(response_column)
    training_columns.remove("UnitNumber")
    training_columns.remove("Time")

    result_list = []
    for tree in tree_arr:
        for depth in depth_arr:
            model = H2ORandomForestEstimator(ntrees=tree, max_depth=depth, nbins=100, seed=12345)
            model.train(x=training_columns, y=response_column, training_frame=train)
            performance = model.model_performance(test_data=test)
            job_no += 1
            result = RandomForestResult(job_no=job_no, mse=model.mse(), mae=model.mae(), ntrees=tree, depth=depth)
            result_list.append(result)
            result.save_to_file()
            Progress.printProgress(iteration=job_no, total=total_work, prefix="Progress", suffix="Complete", barLength=100)

    min_mse = result_list[0].get_result()['mse']
    min_job = result_list[0].get_result()['job_no']
    for result in result_list:
        if result.get_result()['mse'] < min_mse:
            min_mse = result.get_result()['mse']
            min_job = result.get_result()['job_no']

    print "Min mse : " + str(min_mse)
    print "At job  : " + str(min_job)






