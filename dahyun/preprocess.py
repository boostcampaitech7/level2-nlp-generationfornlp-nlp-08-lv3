import re
import pandas as pd
import ast

input_file_path = '/Users/idahyeon/Downloads/nlp_project4/train.csv'
df = pd.read_csv(input_file_path) 

def clean_text(text):
    text = text.replace('\xa0', '') 
    text = re.sub(r'\s+', ' ', text) 
    text = text.replace('...', '')  
    text = text.replace('. . .', '')
    text = text.replace('. .', '')
    text = text.replace('…', '')
    text = text.replace(' .', '.')
    return text.strip()


def clean_list_text(list_text):
    return [clean_text(item) for item in list_text]

def clean_column(value):
    if isinstance(value, str):
        try:
            parsed_value = ast.literal_eval(value)
            if isinstance(parsed_value, dict):

                for key in parsed_value:
                    if isinstance(parsed_value[key], str):
                        parsed_value[key] = clean_text(parsed_value[key])
                    elif isinstance(parsed_value[key], list):
                        parsed_value[key] = clean_list_text(parsed_value[key])
                return parsed_value
            else:
                return clean_text(value)
        except (ValueError, SyntaxError):
            return clean_text(value)
    else:
        return value

df = df.applymap(clean_column)

output_file_path = 'train_cleaned.csv'

df.to_csv(output_file_path, index=False)

