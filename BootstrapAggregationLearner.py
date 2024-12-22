""""""
"""  		  	   		 	   		  		  		    	 		 		   		 		  
An implementation of a Bootstrap Aggregation Learner.  	   		 	   		  		  		    	 		 		   		 		  
"""

import numpy as np
from scipy import stats


class BootstrapAggregationLearner(object):
    """
    This is a Bootstrap Aggregation Learner.

    :param learner: Points to any arbitrary learner class that will be used in the BootstrapAggregationLearner.
    :type learner: learner
    :param kwargs: Keyword arguments that are passed on to the learner’s constructor and they can vary according to the learner
    :param bags: The number of learners you should train using Bootstrap Aggregation.
    :type bags: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, learner, kwargs, bags = 20, boost = False, verbose=False):
        """
        Constructor method
        """
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

        self.learners = []
        for i in range(self.bags):
            self.learners.append(learner(**kwargs))

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner. Perform sampling with replacement.
        If the training set contains n examples each bag should contain n examples as well.

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        for learner in self.learners:
            rand_sample = np.random.randint(0, data_x.shape[0], data_x.shape[0])
            learner.add_evidence(data_x[rand_sample], data_y[rand_sample])

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        result = []
        for learner in self.learners:
            result.append(learner.query(points))
        # Change from regression to classification learner
        # return np.mean(result, axis=0)

        result = np.array(result)
        mode_result = stats.mode(result)

        return mode_result[0][0]