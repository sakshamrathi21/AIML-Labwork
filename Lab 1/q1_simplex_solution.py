"""
Lab week 1: Question 1: Linear programs

Implement solvers to solve linear programs of the form:

max c^{T}x
subject to:
Ax <= b
x >= 0

(a) Firstly, implement simplex method covered in class from scratch to solve the LP

simplex reference:
https://www.youtube.com/watch?v=t0NkCDigq88
"""
import numpy
import pulp
import pandas as pd
import argparse


def parse_commandline_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--testDirectory', type=str, required=True, help='Directory of the test case files')
    arguments = parser.parse_args()
    return arguments


def simplex_solver(A_matrix: numpy.array, c: numpy.array, b: numpy.array) -> list:
    """
    Implement LP solver using simplex method.

    :param A_matrix: Matrix A from the standard form of LP
    :param c: Vector c from the standard form of LP
    :param b: Vector b from the standard form of LP
    :return: list of pivot values simplex method encountered in the same order
    """
    pivot_value_list = []
    ################################################################
    # Implement here
    # Initialize number of basic, non-basic variables
    num_basic_vars = len(b)
    num_non_basic_vars = len(c)
    variables_mapping = {i: i + num_non_basic_vars for i in range(num_basic_vars)}

    # Set tableau_curr i.e. (start with a feasible corner and set initial tableau values)
    tableau_curr = numpy.array(
        [[0.0 for j in range(num_non_basic_vars + num_basic_vars + 1)] for i in range(num_basic_vars + 1)])
    for i in range(num_basic_vars + 1):
        for j in range(num_non_basic_vars + num_basic_vars + 1):
            if j < num_non_basic_vars:
                if i < num_basic_vars:
                    tableau_curr[i, j] = A_matrix[i, j]
                else:
                    tableau_curr[i, j] = -c[j]
            elif num_non_basic_vars <= j < num_non_basic_vars + num_basic_vars:
                if i < num_basic_vars:
                    if i == j - num_non_basic_vars:
                        tableau_curr[i, j] = 1
                    else:
                        tableau_curr[i, j] = 0
                else:
                    tableau_curr[i, j] = 0
            else:
                if i < num_basic_vars:
                    tableau_curr[i, j] = b[i]
                else:
                    tableau_curr[i, j] = 0
    # keep generating tableau until optimal corner point is reached
    while check_obj_row_negative(tableau_curr, num_basic_vars, num_non_basic_vars):
        tableau_next, basic_variables_mapping, pivot_value = calc_next_tableau(tableau_curr, num_basic_vars,
                                                                               num_non_basic_vars,
                                                                               variables_mapping)
        tableau_curr = tableau_next
        pivot_value_list.append(pivot_value)

    # Get the optimal value and x* just for completeness
    x = numpy.array([0.0 for i in range(len(c))])
    for key, val in variables_mapping.items():
        if val < len(c):
            x[val] = tableau_curr[key, num_non_basic_vars + num_basic_vars]
    obj_val = tableau_curr[-1, -1]
    ################################################################

    # Transfer your pivot values to pivot_value_list variable and return
    return pivot_value_list


def calc_next_tableau(tableau_curr, num_basic_vars, num_non_basic_vars, variable_mapping):
    tableau_next = tableau_curr.copy()
    # get pivot row and column
    pivot_column = identify_pivot_column(tableau_curr, num_basic_vars, num_non_basic_vars)
    pivot_row = identify_pivot_row(tableau_curr, pivot_column)
    # Using row ops bring the pivot value to 1 and other pivot column values to 0
    tableau_next[pivot_row, :] = numpy.divide(tableau_curr[pivot_row, :], tableau_curr[pivot_row, pivot_column])
    for i in range(num_basic_vars + 1):
        if i != pivot_row:
            if tableau_curr[i, pivot_column] != 0:
                tableau_next[i, :] = numpy.subtract(tableau_curr[i, :], numpy.multiply(tableau_next[pivot_row, :],
                                                                                       tableau_curr[i, pivot_column]))
    # update the switch in basic variable
    variable_mapping[pivot_row] = pivot_column
    # pivot_value
    pivot_value = tableau_curr[pivot_row, pivot_column]
    return tableau_next, variable_mapping, pivot_value


def check_obj_row_negative(tableau_curr, num_basic_vars, num_non_basic_vars):
    for j in range(num_non_basic_vars + num_basic_vars):
        if tableau_curr[num_basic_vars, j] < 0:
            return True
    return False


def identify_pivot_column(tableau_curr, num_basic_vars, num_non_basic_vars):
    index = 0
    min_val = 0
    for j in range(num_non_basic_vars + num_basic_vars):
        if tableau_curr[num_basic_vars, j] < min_val:
            index = j
            min_val = tableau_curr[num_basic_vars, j]
    return index


def identify_pivot_row(tableau_curr, pivot_column):
    theta = numpy.divide(tableau_curr[:-1, -1], tableau_curr[:-1, pivot_column])
    theta_positive_index = numpy.where(theta >= 0)[0]
    return theta_positive_index[theta[theta_positive_index].argmin()]


if __name__ == "__main__":
    # get command line args
    args = parse_commandline_args()
    if args.testDirectory is None:
        raise ValueError("No file provided")
    # Read the inputs A, b, c and run solvers
    # There are 2 test cases provided to test your code, provide appropriate command line args to test different cases.
    matrix_A = pd.read_csv("{}/A.csv".format(args.testDirectory), header=None, dtype=float).to_numpy()
    vector_c = pd.read_csv("{}/c.csv".format(args.testDirectory), header=None, dtype=float)[0].to_numpy()
    vector_b = pd.read_csv("{}/b.csv".format(args.testDirectory), header=None, dtype=float)[0].to_numpy()

    simplex_pivot_values = simplex_solver(matrix_A, vector_c, vector_b)
    for val in simplex_pivot_values:
        print(val)
