import pandas as pd
import ast
import random

# Load the provided CSV file
file_path = 'need_knowledge.csv'
df = pd.read_csv(file_path)

# Convert 'problems' column from string representation of dictionary to actual dictionary
df['problems'] = df['problems'].apply(ast.literal_eval)

# Make a deep copy of the dataframe to create the shuffled version
shuffled_df = df.copy(deep=True)

# Iterate through each problem and shuffle the choices
for idx, problem in shuffled_df['problems'].items():
    original_choices = problem['choices']
    correct_answer = problem['answer'] - 1  # Convert to 0-based index
    
    while True:
        shuffled_choices = original_choices[:]
        random.shuffle(shuffled_choices)
        if shuffled_choices[correct_answer - 1] != original_choices[correct_answer - 1]:  # Ensure the order has changed
            break

    # Find the new index of the correct answer
    new_answer = shuffled_choices.index(original_choices[correct_answer]) + 1  # Convert back to 1-based index
    
    # Update the problem dictionary with shuffled choices and new answer
    problem['choices'] = shuffled_choices
    problem['answer'] = new_answer
    shuffled_df.at[idx, 'problems'] = problem

# Add a new column 'id' to shuffled_df with the original id plus '_shuffle'
shuffled_df['id'] = df['id'].astype(str) + '_shuffle'

# Concatenate the original dataframe with the shuffled dataframe
combined_df = pd.concat([df, shuffled_df], ignore_index=True)

# Save the combined dataframe to a new CSV file
output_file_path = 'shuffle_data.csv'
combined_df.to_csv(output_file_path, index=False)

print("Shuffled and combined data saved to", output_file_path)
