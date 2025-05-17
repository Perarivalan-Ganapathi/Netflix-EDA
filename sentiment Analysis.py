from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:\\Users\\perar\\Desktop\\Netflix_movies_and_tv_shows_clustering.csv\\Cleaned_Netflix_Data.csv")

def sentiment_analyzer(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df['sentiment'] = df['description'].apply(sentiment_analyzer)
df['label'] = pd.cut(df['sentiment'], bins=[-1, -0.1, 0.1, 1], labels=['Negative', 'Neutral', 'Positive'])

plt.title('Sentimental Analysis')
sns.countplot(x='label', data=df, palette='Set2')
plt.show()