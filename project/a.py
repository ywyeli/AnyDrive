###

import pandas as pd
import numpy as np

# Load the uploaded Excel file
file_path = '/home/ye/Downloads/T1.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the data to understand its structure
data.head()

# Generate random noise within the range of ±2 for each element in the dataframe
noise = np.random.uniform(2, 4, size=data.shape)

# Add the noise to the original data
data_noisy = data + noise

# Display the first few rows of the modified data to verify the addition of noise
# data_noisy.head()

df = pd.DataFrame(data_noisy)

# Specify the file name
file_name = 'T1_2_output.csv'

# Save the DataFrame to a CSV file
df.to_csv(file_name, index=False)
