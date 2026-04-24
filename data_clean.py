import pandas as pd

df = pd.read_csv("test_features_dataset.csv")
df2 = pd.read_csv("test_data.csv")
print(df2.head())

df = df.drop(columns=["image_id", "file_name", "folder_name", "food_name", "is_fruit", "file_ext", "height", "width"])
df.to_csv("test_data.csv", index=False)