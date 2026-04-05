import pandas as pd

# Load the file
df = pd.read_csv("RecipeNLG_train_ready_60k.csv")

# Remove the columns
df = df.drop(columns=["input_length", "output_length"], errors="ignore")

# Save as final dataset
df.to_csv("final_dataset.csv", index=False)

print("File cleaned and saved as final_dataset.csv")