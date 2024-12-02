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
    # %% Student Code Start
    # Implement here
    simplex_table = A_matrix
    rows, columns = numpy.shape(A_matrix)
    identity_matrix = numpy.identity(columns, dtype="float")
    simplex_table = numpy.concatenate((simplex_table, identity_matrix), axis=1)
    # print(numpy.shape(simplex_table))
    simplex_table = numpy.c_[simplex_table, b]
    rows, columns = numpy.shape(identity_matrix)
    zero_array = numpy.zeros((rows+1))
    lower_row = numpy.concatenate((-numpy.transpose(c), zero_array))
    # numpy.append
    simplex_table = numpy.r_[simplex_table, [lower_row]]
    # print(simplex_table)
    rows, columns = numpy.shape(simplex_table)
    while(1):
        max_column = 0
        for i in range(columns-1):
            # print(simplex_table[rows-1][i])
            if simplex_table[rows-1][i] < simplex_table[rows-1][max_column]:
                # print(max_column, i)
                max_column = i
        if simplex_table[rows-1][max_column] >= 0:
            break
        pivot_row = 0
        while(simplex_table[pivot_row][max_column] == 0):
            pivot_row = pivot_row + 1

        while (simplex_table[pivot_row][columns-1]/simplex_table[pivot_row][max_column]<0):
            pivot_row = pivot_row + 1
        # print(pivot_row)
        if pivot_row >= rows-1:
            return pivot_value_list
        for i in range(rows-1):
            
            if (simplex_table[i][max_column]!=0):
                if simplex_table[i][columns-1]/simplex_table[i][max_column] > 0:
                    # print(simplex_table[i][columns-1]/simplex_table[i][max_column])
                    # print(simplex_table[pivot_row][columns-1]/simplex_table[pivot_row][max_column])
                    if simplex_table[i][columns-1]/simplex_table[i][max_column] <= simplex_table[pivot_row][columns-1]/simplex_table[pivot_row][max_column]:
                        # print("hello")
                        pivot_row = i
        # print(pivot_row, max_column)
        # break
        # print(simplex_table)
        pivot_value_list.append(simplex_table[pivot_row][max_column])
        divisor = simplex_table[pivot_row][max_column]
        for i in range(columns):
            # print(simplex_table[pivot_row][i])
            simplex_table[pivot_row][i] = simplex_table[pivot_row][i]/divisor
            # print(simplex_table[pivot_row][i])
        # print(simplex_table)
        for i in range(rows):
            if i == pivot_row:
                continue
            current_divisor = simplex_table[i][max_column]/simplex_table[pivot_row][max_column]
            # print(current_divisor)
            for j in range(columns):
                simplex_table[i][j] = simplex_table[i][j] - current_divisor*simplex_table[pivot_row][j]
        # print(simplex_table)
        # break

     # %% Student Code End
    ################################################################

    # Transfer your pivot values to pivot_value_list variable and return
    return pivot_value_list


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
