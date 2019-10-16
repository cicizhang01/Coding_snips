import random
import numpy as np

def generate_target_function():
    """
    Generates random target function with d = 2 by taking two random points between
    [-1, 1] x [-1, 1] and creating a random line
    """
    x1, y1 = random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)
    x2, y2 = random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)
    m = (y2 - y1) / (x2 - x1)
    b = y2 - m * x1
    return m, b

def generate_data(num_training_pts, m, b):
    """
    Creates data set. Inputs are two dim and Output is either -1 or 1. Point is -1 if below target function line
    and 1 if above.
    Returns: [[x, y], output]
    """
    training_data = []
    for i in range(num_training_pts):
        x, y = random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)
        comparison = x * m + b
        if y > comparison:
            output = 1
        else:
            output = -1
        training_data.append([[x, y], output])
    return training_data

def calculate_predictions(weight, point):
    """
    Calculates predicted output using given weights
    """
    point = [1] + point
    numpy_point = np.array(point)
    sign = np.sign(np.dot(weight, numpy_point))
    return sign

def get_miscalculated_points(weight, data):
    """
    Returns array with points that are misclassified using predicted weights
    """
    miscalculated_pts = []
    for point in data:
        pred = calculate_predictions(weight, point[0])
        if pred != point[1]:
            miscalculated_pts.append(point)
    return miscalculated_pts

def fix_weights(weights, point):
    """
    Adjusts weights according to misclassified point
    """
    coord = [1] + point[0]
    numpy_point = np.array(coord)
    new_weight = weights + numpy_point * point[1]
    return new_weight

def run_trials(num_points, num_of_times):
    """
    Each run (there are num_of_times runs) chooses a random line in X = [-1, 1] x [-1, 1].
    Data set of size num_points is randomly generated. Each run uses Perceptron Learning
    Algorithm to predict target function (randomly generated line).

    Prints:
        avg_iters : average number of iterations of PLA for prediction to converge
        avg_prob_failing : average probability that target function and prediction
            function will disagree on classification of point
    """
    total_iters = 0
    prob_failing = 0
    for i in range(num_of_times):
        m, b = generate_target_function()
        training_data = generate_data(num_points, m, b)
        weights = np.array([0.0, 0.0, 0.0])

        mis_points = get_miscalculated_points(weights, training_data)
        iteration = 0
        while(True):
            if len(mis_points) > 0:
                bad_point = random.choice(mis_points)
                weights = fix_weights(weights, bad_point)
                mis_points = get_miscalculated_points(weights, training_data)
                iteration += 1
            else:
                break
        total_iters += iteration

        testing_data = generate_data(1000, m, b)
        failed_testing_data = get_miscalculated_points(weights, testing_data)
        prob_failing += len(failed_testing_data) / 1000
    avg_prob_failing = prob_failing / num_of_times
    avg_iters = total_iters / num_of_times
    print("Average iters: {} Average probability to fail: {}".format(avg_iters, avg_prob_failing))

run_trials(10, 1000)
run_trials(100, 1000)
