import re
import pandas as pd
import ast

input_file_path = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/sungeun/data/new_train3_cleaned286.csv'
df = pd.read_csv(input_file_path) 

def clean_text(text):
    #text = text.replace('\xa0', '') 
    #text = re.sub(r' {2,}', ' ', text)
    # text = text.replace('...', '')  
    # text = text.replace('. . .', '')
    # text = text.replace('. .', '')
    text = text.replace('…', '')
    # text = text.replace(' .', '.')
    return text.strip()


def clean_list_text(list_text):
    cleaned_list = [clean_text(item) for item in list_text]
    if len(cleaned_list) == 4:
        cleaned_list.append(' ')
    return cleaned_list

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

output_file_path = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/dahyun/data/new_train.csv'

df.to_csv(output_file_path, index=False)

