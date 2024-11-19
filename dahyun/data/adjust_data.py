import pandas as pd
import ast
import random
from collections import Counter

# Load the provided CSV file
file_path = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/new_train.csv'
df = pd.read_csv(file_path)

# Convert 'problems' column from string representation of dictionary to actual dictionary
df['problems'] = df['problems'].apply(ast.literal_eval)

# Calculate the current distribution of answers
answer_counts = df['problems'].apply(lambda x: x['answer'])
answer_distribution = Counter(answer_counts)

# Calculate the target count for each answer to be evenly distributed
total_answers = len(df)
num_choices = 5
target_count = total_answers // num_choices

# Separate the indices of each answer category
indices_by_answer = {i: [] for i in range(1, num_choices + 1)}
for idx, problem in enumerate(df['problems']):
    indices_by_answer[problem['answer']].append(idx)

# Create a new list of indices to adjust
indices_to_adjust = []
for answer, indices in indices_by_answer.items():
    if len(indices) > target_count:
        indices_to_adjust.extend(indices[target_count:])

# Shuffle the indices to ensure a random selection
random.shuffle(indices_to_adjust)

# Adjust answers to balance the distribution
for idx in indices_to_adjust:
    current_answer = df.at[idx, 'problems']['answer']
    # Find a new answer that is under the target count
    for new_answer in range(1, num_choices + 1):
        if len(indices_by_answer[new_answer]) < target_count:
            # Update the answer
            df.at[idx, 'problems']['answer'] = new_answer

            # Update the choices accordingly (swap the original answer with the new one)
            original_choices = df.at[idx, 'problems']['choices']
            original_index = current_answer - 1
            new_index = new_answer - 1
            updated_choices = original_choices[:]
            updated_choices[original_index], updated_choices[new_index] = (
                updated_choices[new_index],
                updated_choices[original_index],
            )
            df.at[idx, 'problems']['choices'] = updated_choices

            # Update the indices_by_answer dictionary
            indices_by_answer[current_answer].remove(idx)
            indices_by_answer[new_answer].append(idx)
            break

# Save the updated dataframe to a new CSV file
output_file_path = 'adjusted_data.csv'
df.to_csv(output_file_path, index=False)