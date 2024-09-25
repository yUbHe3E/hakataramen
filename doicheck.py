import os
import pandas as pd
import re

def extract_data_from_file(file_path):
    with open(file_path, 'r') as file:
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
        return pd.DataFrame()  # Skip files that do not match the expected format

    zeolite_type, adsorbate, temperature = match.groups()

    # Create a DataFrame with the necessary columns
    df_data = pd.DataFrame({
        "zeolite_type": [zeolite_type],
        "adsorbate": [adsorbate],
        "temperature": [temperature],
        "doi": [doi]
    })

    return df_data

def process_files_in_directory(directory_path, output_file):
    all_data = []

    # Walk through the directory and process each CSV file
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = extract_data_from_file(file_path)
                if not df.empty:
                    all_data.append(df)

    # Combine all dataframes into a single dataframe
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Data successfully extracted to {output_file}")
    else:
        print("No valid data found in the directory.")

# Example usage
directory_path = './zeoliteNISTdata/'  # Replace with the path to your directory
output_file = 'doicheck.csv'  # Output file name
process_files_in_directory(directory_path, output_file)
