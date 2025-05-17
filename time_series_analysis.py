import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sna

df=pd.read_csv("C:\\Users\\perar\\Desktop\\Netflix_movies_and_tv_shows_clustering.csv\\Cleaned_Netflix_Data.csv")

df['date_added'] = pd.to_datetime(df['date_added'])

df['year_added'] = df['date_added'].dt.year

yearly_content = df.groupby(['year_added', 'type']).size().unstack()

print(yearly_content)

yearly_content.plot(kind='line', marker='o', ax=plt.gca())
plt.title('Yearly Trend of Content Addition on Netflix')
plt.xlabel('Year')
plt.ylabel('Number of Titles')
plt.grid(True)
plt.show()