import pandas as pd

from processor_regex import classify_with_regex
from processor_bert import classify_with_bert
from processor_llm import classify_with_llm


def classify_log(source, log_msg):
    if source == 'LegacyCRM':
        label = classify_with_llm(log_msg)
    else:
        label = classify_with_regex(log_msg)
        if not label:
            label = classify_with_bert(log_msg)
    return label

def classify(logs):
    labels = []
    for source, log_msg in logs:
        label = classify_log(source, log_msg)
        labels.append(label)
    return labels

def classify_csv(file_path):
    df = pd.read_csv(file_path)

    df['target_label'] = classify(list(zip(df['source'], df['log_message'])))

    df.to_csv('test.csv', index=False)
    print(f"Classified logs saved to test.csv")


if __name__ == "__main__":
    classify_csv("testing/test.csv")