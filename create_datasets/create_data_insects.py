from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 2023

# Load ARFF file
data, meta = arff.loadarff('InsectSound.arff')

# Convert to a Pandas DataFrame if needed
df = pd.DataFrame(data)

targets = df['target'].unique()
target_values_dict = {}
for i, name in enumerate(targets):
    target_values_dict[name] = i
    
df["target"] = df["target"].replace(target_values_dict)

# Split the data
df_train, df_test = train_test_split(df, test_size = 0.2, random_state = SEED)

print(f"train: {len(df_train)} | test: {len(df_test)}\n")
print(f"{df_test['target'].value_counts()}")

df_train.to_csv("train_insects.csv",sep=",",index=False)
df_test.to_csv("test_insects.csv",sep=",",index=False)

