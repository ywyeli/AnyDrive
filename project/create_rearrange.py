start_sequence = 0
end_sequence = 10

# Define the base filename pattern
base_filename = ''

# Function to generate the filenames
def generate_filenames(start, end, per_line=70):
    # filenames = [f"{base_filename}{i:02d}" for i in range(start, end)]
    filenames = [f'{i + j}' for i in range(start, end) for j in range(0, 70, 10)]
    formatted_filenames = [filenames[i:i+per_line] for i in range(0, len(filenames), per_line)]
    return formatted_filenames

# Generate the filenames with new lines every 8 names
formatted_filenames = generate_filenames(start_sequence, end_sequence)

# Save the filenames to a text file
file_path = 'routes_names.txt'
with open(file_path, 'w') as file:
    for filenames_line in formatted_filenames:
        file.write(', '.join(filename for filename in filenames_line) + ',\n')

print(f"Filenames generated and saved to '{file_path}'.")