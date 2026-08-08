"""
Better handling of missing values in feature matrix
"""
import pandas as pd
import numpy as np

# Load the feature matrix
df = pd.read_csv('data/processed/manga_features.csv')

print("="*70)
print("HANDLING MISSING VALUES")
print("="*70)

print("\nBefore cleaning:")
print(f"Total NaN values: {df.isnull().sum().sum()}")
print(f"Percentage: {df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.1f}%")

# Show which columns have NaNs
nan_by_column = df.isnull().sum()
nan_cols = nan_by_column[nan_by_column > 0].sort_values(ascending=False)

if len(nan_cols) > 0:
    print(f"\nColumns with missing values ({len(nan_cols)} total):")
    for col, count in nan_cols.items():
        pct = count / len(df) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")

numeric_cols = df.select_dtypes(include=[np.number]).columns

# Drop columns with more than 50% missing
cols_to_drop = []
for col in numeric_cols:
    if col == 'manga_id':
        continue
    missing_pct = df[col].isnull().sum() / len(df) * 100
    if missing_pct > 50:
        cols_to_drop.append(col)
        print(f"\nDropping '{col}' ({missing_pct:.1f}% missing)")

df = df.drop(columns=cols_to_drop)

# Fill remaining NaNs with median for numeric columns
for col in numeric_cols:
    if col in df.columns and col != 'manga_id':
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

# Fill NaNs in binary features (genre_, demographic_) with 0
binary_cols = [col for col in df.columns if col.startswith('genre_') or col.startswith('demographic_')]
for col in binary_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(0, inplace=True)

print("\n" + "="*70)
print("After cleaning:")
print(f"Total NaN values: {df.isnull().sum().sum()}")
print(f"Shape: {df.shape}")

if df.isnull().sum().sum() == 0:
    print("All missing values handled!")
else:
    print("Some NaNs remain:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

# Save cleaned features
df.to_csv('data/processed/manga_features.csv', index=False)
print(f"\nSaved cleaned features to data/processed/manga_features.csv")

# Show final summary
print("\nFinal Feature Summary:")
print(df.describe())
