"""
Lab week 1: Question 1: Linear programs

Implement solvers to solve linear programs of the form:

max c^{T}x
subject to:
Ax <= b
x >= 0

(b) Secondly, make use pulp package utilities to solve the LP.

pulp references:
(1) https://coin-or.github.io/pulp/main/includeme.html#examples
(2) https://coin-or.github.io/pulp/technical/pulp.html
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


def pulp_solver(A_matrix: numpy.array, c: numpy.array, b: numpy.array) -> (numpy.array, float):
    """
    Implement LP solver using pulp utilities.

    :param A_matrix: Matrix A from the standard form of LP
    :param c: Vector c from the standard form of LP
    :param b: Vector b from the standard form of LP
    :return: (numpy.array, float) return the solution x* and optimal value
    """
    x = numpy.array([0.0 for i in range(len(c))])
    opt_val = 0.0
    ################################################################
    # Implement here
    # variables
    num_variables = len(c)
    variables = numpy.array([pulp.LpVariable("{}".format(i), lowBound=0) for i in range(num_variables)])

    # problem + objective function
    problem = pulp.LpProblem("pulp_solver", pulp.LpMaximize)
    objective_func = (
        pulp.LpAffineExpression([(variables[i], c[i]) for i in range(num_variables)]), "objective function")
    problem += objective_func

    # constraints
    num_constraints = len(b)
    for i in range(num_constraints):
        problem += pulp.LpConstraint(
            pulp.LpAffineExpression([(variables[j], A_matrix[i, j]) for j in range(num_variables)]),
            pulp.LpConstraintLE, "constraint_{}".format(i), b[i])

    # solve
    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    # Transfer solution to a numpy array
    x = numpy.array([pulp.value(variables[i]) for i in range(num_variables)])
    opt_val = problem.objective.value()
    ################################################################

    # Transfer your solution to x and opt_val and finally return the x vector i.e. solution (numpy array) and the
    # optimal objective function value (float value)
    return x, opt_val


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

    x_pulp, obj_val_pulp = pulp_solver(matrix_A, vector_c, vector_b)
    for val in x_pulp:
        print(val)
    print(obj_val_pulp)
