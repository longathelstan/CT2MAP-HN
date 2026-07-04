import pandas as pd
df = pd.read_csv('data/processed/manifests/manifest.csv')
print(df['split'].value_counts())
