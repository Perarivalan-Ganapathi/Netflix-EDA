import pandas as pd


df=pd.read_csv("Netflix_movies_and_tv_shows_clustering.csv")

# Fill the Missing values with Unknown or NA
df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')
df['date_added'] = df['date_added'].fillna('Unknown')
df['rating'] = df['rating'].fillna('Not Rated')

# Change the Date format to datetime
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Remove Duplicates
df = df.drop_duplicates()

# Data Standardization - Removing leading and trailing spaces from string columns
cols=['show_id','type','title','director','cast','country','rating','duration',
     'listed_in','description']
for c in cols:
    df[c] = df[c].str.strip().str.lower()

# checking the dataset to ensure the updates are correct
print(df.info())

# Saving cleaned data to a new file
df.to_csv('cleaned_netflix_data.csv', index=False)

