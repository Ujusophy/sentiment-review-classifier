import pandas as pd

def load_file(path):
    return pd.read_csv(path, sep='\t', names=["text", "label"], header=None)

# Load each file
amazon = load_file("amazon_cells_labelled.txt")
imdb   = load_file("imdb_labelled.txt")
yelp   = load_file("yelp_labelled.txt")

# Combine
df = pd.concat([amazon, imdb, yelp], ignore_index=True)

# Save as CSV
df.to_csv("sentiment_dataset.csv", index=False)
print("✅ Dataset saved as sentiment_dataset.csv")