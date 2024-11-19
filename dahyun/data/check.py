import pandas as pd
import ast

# Load the original and updated CSV files
original_file_path = 'new_train.csv'
updated_file_path = 'adjusted_data.csv'

# Read the CSV files into dataframes
df_original = pd.read_csv(original_file_path)
df_updated = pd.read_csv(updated_file_path)

# Convert 'problems' column from string representation of dictionary to actual dictionary
df_original['problems'] = df_original['problems'].apply(ast.literal_eval)
df_updated['problems'] = df_updated['problems'].apply(ast.literal_eval)

# Iterate over both dataframes to compare answer choices
mismatches = []
for idx in range(len(df_original)):
    original_problem = df_original.at[idx, 'problems']
    updated_problem = df_updated.at[idx, 'problems']

    original_answer = original_problem['answer']
    updated_answer = updated_problem['answer']

    original_choices = original_problem['choices']
    updated_choices = updated_problem['choices']

    # Compare the answer values and their respective choices
    if original_answer != updated_answer:
        original_answer_value = original_choices[original_answer - 1]
        updated_answer_value = updated_choices[updated_answer - 1]

        if original_answer_value != updated_answer_value:
            mismatches.append({
                'index': idx,
                'original_answer': original_answer,
                'updated_answer': updated_answer,
                'original_answer_value': original_answer_value,
                'updated_answer_value': updated_answer_value
            })

# Display mismatches, if any
if mismatches:
    for mismatch in mismatches:
        print(f"Index {mismatch['index']}: Original answer value '{mismatch['original_answer_value']}' does not match updated answer value '{mismatch['updated_answer_value']}'")
else:
    print("All answer choices match between the original and updated files.")
