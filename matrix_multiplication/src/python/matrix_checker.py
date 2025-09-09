import numpy as np
import sys
from pathlib import Path

def read_csv_to_matrix(filename):
    try:
        # loadtxt is a simple and efficient way to load numerical data [6, 9]
        matrix = np.loadtxt(filename, delimiter=',')
        return matrix
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading '{filename}': {e}")
        return None

def multiply_matrices(matrix_a, matrix_b):
    try:
        # The @ operator is a convenient way to perform matrix multiplication in NumPy
        result_matrix = matrix_a @ matrix_b
        return result_matrix
    except ValueError as e:
        print(f"Error: Matrix multiplication is not possible. {e}")
        print(f"Please check the dimensions: Matrix A has {matrix_a.shape[1]} columns and Matrix B has {matrix_b.shape[0]} rows.")
        return None

def compare_matrices(matrix_a, matrix_b):
    # np.allclose is recommended for comparing floating-point matrices as it handles small precision differences [5]
    return np.allclose(matrix_a, matrix_b)

def main():
    script_dir = Path(__file__).resolve().parent 
    mm_folder = script_dir.parent.parent
    data_path = mm_folder / 'data' / 'sample_input_matrices'

    # Define the filenames for the matrices
    matrix_a_file = data_path / 'matrix_a.csv'
    matrix_b_file = data_path / 'matrix_b.csv'
    mpi_result_file = data_path / 'matrix_c.csv'

    print("--- Reading Matrices from CSV Files ---")
    matrix_a = read_csv_to_matrix(matrix_a_file)
    matrix_b = read_csv_to_matrix(matrix_b_file)
    matrix_mpi = read_csv_to_matrix(mpi_result_file)

    # Check if all files were read successfully
    if matrix_a is None or matrix_b is None or matrix_mpi is None:
        sys.exit("Exiting due to file reading errors.")

    print("\nMatrix A:")
    print(matrix_a)
    print("\nMatrix B:")
    print(matrix_b)
    print("\nMpi calculated matrix:")
    print(matrix_mpi)

    print("\n--- Performing Matrix Multiplication (A * B) ---")
    result_matrix = multiply_matrices(matrix_a, matrix_b)

    if result_matrix is not None:
        print("\nCalculated Result Matrix:")
        print(result_matrix)

        print("\n--- Comparing Calculated Result with Expected Result ---")
        are_equal = compare_matrices(result_matrix, matrix_mpi)

        if are_equal:
            print("\n✅ Success: The calculated matrix is the same as the expected matrix.")
        else:
            print("\n❌ Failure: The calculated matrix is DIFFERENT from the expected matrix.")
            # Optional: show the difference
            difference = result_matrix - matrix_mpi
            print("\nDifference (Calculated - Expected):")
            print(difference)

if __name__ == "__main__":
    main()