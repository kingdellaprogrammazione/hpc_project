import numpy as np
import csv
from pathlib import Path
import sys

# Define the dimension of the square matrices
if len(sys.argv) > 2:
    print("Usage: python3 matrix_generator.py <side>")
    sys.exit(1)

if (len(sys.argv) == 1):
    side_dimension = 50  # You can change this value to any size you need
else:
    side_dimension = sys.argv[1]

side_dimension = int(side_dimension)

script_dir = Path(__file__).resolve().parent 
mm_folder = script_dir.parent.parent
data_path = mm_folder / 'data' / 'sample_input_matrices'

# Define the names for the output files
file_a = data_path / 'matrix_a.csv'
file_b = data_path / 'matrix_b.csv'
file_c = data_path / 'matrix_c.csv'

# --- 1. Generate and save the first matrix (e.g., with floating-point numbers) ---
print(f"Generating a {side_dimension}x{side_dimension} matrix of floats...")
# Create a matrix with random float values between 0 and 1
matrix_a = np.round(np.random.rand(side_dimension, side_dimension), decimals=2)

# Write matrix_a to its CSV file
with open(file_a, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(matrix_a)
print(f"'{file_a}' created successfully.")


# --- 2. Generate and save the second matrix (e.g., with integers) ---
print(f"Generating a {side_dimension}x{side_dimension} matrix of integers...")
# Create a matrix with random integer values between 0 and 99
matrix_b = np.random.randint(100, size=(side_dimension, side_dimension))

# Write matrix_b to its CSV file
with open(file_b, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(matrix_b)
print(f"'{file_b}' created successfully.")


# --- 3. Create the third, empty file for the result ---
print("Creating an empty file for the results...")
with open(file_c, 'w') as f:
    # Opening the file in write mode ('w') and immediately closing it
    # is enough to create an empty file.
    pass
print(f"'{file_c}' created successfully.")

print("\nAll files have been generated.")