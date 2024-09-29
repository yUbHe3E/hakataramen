import os
import csv
import re

def extract_data_from_file(file_path):
    name = None
    # Read the CSV file using the csv module
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            # Get the value from the second column and the ninth row (row index 8, column index 1)
            if len(rows) > 8 and len(rows[8]) > 1:
                name = rows[8][1]
            else:
                name = "Data not found"
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None  # Return None if there is an error

    # Read the file content for DOI search
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.readlines()

    # Look for DOI within the file content
    doi_line = None
    for line in file_content:
        if "doi" in line.lower():
            doi_line = line.strip()
            break

    if doi_line:
        doi = doi_line.split(',')[1].strip()  # Extract the actual DOI
    else:
        doi = "DOI not found"

    # Extract the zeolite type, adsorbate, and temperature from the filename
    filename = file_path.split('/')[-1]
    match = re.match(r'(\w+)-(\w+)-(\d+)', filename)
    if not match:
        return None  # Skip files that do not match the expected format

    zeolite_type, adsorbate, temperature = match.groups()

    # Return a dictionary with the necessary data
    return {
        "zeolite_type": zeolite_type,
        "adsorbate": adsorbate,
        "temperature": temperature,
        "doi": doi,
        "name": name
    }

def process_files_in_directory(directory_path, output_file):
    all_data = []

    # Walk through the directory and process each CSV file
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                data = extract_data_from_file(file_path)
                if data:
                    all_data.append(data)

    # Write the combined data to an output CSV file
    if all_data:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["zeolite_type", "adsorbate", "temperature", "doi", "name"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_data)
        print(f"Data successfully extracted to {output_file}")
    else:
        print("No valid data found in the directory.")

# Example usage
directory_path = './zeoliteNISTdata/'  # Replace with the path to your directory
output_file = 'doicheck.csv'  # Output file name
process_files_in_directory(directory_path, output_file)
