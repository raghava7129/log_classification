import re
import pandas as pd
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer


# Code used to identify this patterns
def check_patterns():
    df = pd.read_csv("training/dataset/training_logs.csv")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(
        df['log_message'].tolist(), batch_size=64, show_progress_bar=True
    )

    clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(embeddings)
    df['cluster'] = clustering.labels_
    clusters = df.groupby('cluster')['log_message'].apply(list)
    sorted_clusters = clusters.sort_values(
        key = lambda x: x.map(len),
        ascending = False  
    )

    print("Clustered Patterns:")
    for cluster_id, messages in sorted_clusters.items():
        if len(messages) > 10:
            print(f"Cluster {cluster_id}:")
            for msg in messages[:5]:
                print(f"  {msg}")


# regex processor code.
regex_patterns = {
    r"User User\d+ logged (in|out).": "User Action",
    r"Backup (started|ended) at .*": "System Notification",
    r"Backup completed successfully.": "System Notification",
    r"System updated to version .*": "System Notification",
    r"File .* uploaded successfully by user .*": "System Notification",
    r"Disk cleanup completed successfully.": "System Notification",
    r"System reboot initiated by user .*": "System Notification",
    r"Account with ID .* created by .*": "User Action"
}

def classify_with_regex(log_message):
    for pattern, label in regex_patterns.items():
        if re.search(pattern, log_message, re.IGNORECASE):
            return label
    return None

if __name__ == "__main__":
    # check_patterns()
    print(classify_with_regex("Account with ID 1234 created by raghs."))
    print(classify_with_regex("chat are we cooked?"))