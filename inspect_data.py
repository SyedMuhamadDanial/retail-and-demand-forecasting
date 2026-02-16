import pandas as pd

print("=" * 60)
print("TRAIN.CSV")
print("=" * 60)
train = pd.read_csv('train.csv', nrows=5)
print(f"Columns: {train.columns.tolist()}")
print(f"\nFirst 3 rows:\n{train.head(3)}")
print(f"\nData types:\n{train.dtypes}")

print("\n" + "=" * 60)
print("TEST.CSV")
print("=" * 60)
test = pd.read_csv('test.csv', nrows=5)
print(f"Columns: {test.columns.tolist()}")
print(f"\nFirst 3 rows:\n{test.head(3)}")

print("\n" + "=" * 60)
print("STORE.CSV")
print("=" * 60)
store = pd.read_csv('store.csv')
print(f"Shape: {store.shape}")
print(f"Columns: {store.columns.tolist()}")
print(f"\nAll rows:\n{store}")
