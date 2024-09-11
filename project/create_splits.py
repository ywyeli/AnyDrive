# Define the range of filenames
start_sequence = 1
end_sequence = 250

# Define the base filename pattern
base_filename = 'carla-'

# Function to generate the filenames
def generate_filenames(start, end, per_line=10):
    filenames = [f"{base_filename}{i:04d}" for i in range(start+1000, end + 1001)]
    formatted_filenames = [filenames[i:i+per_line] for i in range(0, len(filenames), per_line)]
    return formatted_filenames

# Generate the filenames with new lines every 8 names
formatted_filenames = generate_filenames(start_sequence, end_sequence)

# Save the filenames to a text file
file_path = 'scene_filenames.txt'
with open(file_path, 'w') as file:
    for filenames_line in formatted_filenames:
        file.write(', '.join(f"'{filename}'" for filename in filenames_line) + ',\n')

print(f"Filenames generated and saved to '{file_path}'.")
