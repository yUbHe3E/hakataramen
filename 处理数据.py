import os
import pandas as pd
import re


def extract_data_from_file(file_path):
    with open(file_path, 'r') as file:
        file_content = file.readlines()

    # Identify the start of the data section
    data_start_index = None
    for idx, line in enumerate(file_content):
        if "line" in line.lower() and "pressure" in line.lower():
            data_start_index = idx
            break

    if data_start_index is None:
        return pd.DataFrame()  # Return an empty dataframe if no data found

    # Extract the zeolite type, adsorbate, and temperature from the filename
    filename = file_path.split('/')[-1]
    match = re.match(r'(\w+)-(\w+)-(\d+)', filename)
    if not match:
        return pd.DataFrame()  # Skip files that do not match the expected format

    zeolite_type, adsorbate, temperature = match.groups()

    # Read the relevant data section into a dataframe
    data_lines = file_content[data_start_index + 1:]
    data = [line.strip().split(',') for line in data_lines]

    # Create a DataFrame with the necessary columns
    df_data = pd.DataFrame(data, columns=["line", "pressure", "composition", "adsorption", "total_adsorption"])
    df_data = df_data[["pressure", "adsorption"]].copy()

    # Add the extracted zeolite type, adsorbate, and temperature to the dataframe
    df_data["zeolite_type"] = zeolite_type
    df_data["adsorbate"] = adsorbate
    df_data["temperature"] = temperature

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
output_file = 'combined_data.csv'  # Output file name
process_files_in_directory(directory_path, output_file)
