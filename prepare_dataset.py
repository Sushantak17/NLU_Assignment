import json
import pandas as pd

sports = []
politics = []

# Read the JSON dataset line by line
with open("data/News_Category_Dataset_v3.json", "r", encoding="utf-8") as f:
    for line in f:
        article = json.loads(line)

        text = article["headline"] + " " + article["short_description"]
        category = article["category"]

        if category == "SPORTS":
            sports.append(text)
        elif category == "POLITICS":
            politics.append(text)

# Take first 1000 samples from each category
sports = sports[:1000]
politics = politics[:1000]

# Save to txt files (one document per line)
pd.Series(sports).to_csv("data/sports.txt", index=False, header=False)
pd.Series(politics).to_csv("data/politics.txt", index=False, header=False)


print("Dataset prepared successfully: sports.txt and politics.txt created")
