import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from difflib import get_close_matches

# Page Configuration
st.set_page_config(page_title="Netflix EDA Dashboard", page_icon="📈", layout="wide")

# Footer function
def footer():
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #111;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            border-top: 1px solid #333;
        }
        </style>
        <div class="footer">
            <p>Created by Perarivalan Ganapathi 
            | 📧 <a href="mailto:perarivalang164@gmail.com" style="color: #f0f0f0;">perarivalang164@gmail.com</a> 
            | 🌐 <a href="https://www.linkedin.com/in/perarivalan-ganapathi?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BsWYYgbyUSQ6l63dAYGMEmQ%3D%3D" style="color: #f0f0f0;">LinkedIn</a>
            | 🐙 <a href="https://github.com/Perarivalan-Ganapathi" style="color: #f0f0f0;">GitHub</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Import dataset

df=pd.read_csv("Cleaned_Netflix_Data.csv")

# Modify the dataset in require format for your work
df['date_added'] = pd.to_datetime(df['date_added'])
df['year'] = df['date_added'].dt.year


# Title and Introduction
st.title("Netflix Data Analysis Dashboard")
st.markdown("An interactive dashboard showcasing insights from Netflix's content library.")

# Recommendation System
st.header("🎥 Enter a title to get relevant Movies/Series")
titl = st.text_input("Enter & click Get Recommendation :").lower()
no_of_recommend=st.slider('Set no of relevant title you want!', 1, 10, 5)

# create a column that contains cast and type of shows
df['combine'] = df['listed_in'] + ' ' + df['cast']

vector = TfidfVectorizer(stop_words='english')
feature = vector.fit_transform(df['combine'])
similarity_matrix = cosine_similarity(feature)

if st.button('Get Recommendations'):
    closest_matches = get_close_matches(titl, df['title'], n=1, cutoff=0.6)
    if closest_matches:
        matched_title = closest_matches[0]
        idx=df[df['title'].str.lower() == matched_title.lower()].index[0]
        similarity_score = list(enumerate(similarity_matrix[idx].flatten()))
        sorted_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)[1:no_of_recommend+1]
        title_recommend = [df.iloc[index]['title'] for index, _ in sorted_score]
        st.write('Recommendation for the title:', titl.title())

        for i, titles in enumerate(title_recommend, 1):
            st.write(f"{i}. {titles.title()}")
    else:
        st.warning("Title not found. Please enter a valid Netflix title.")

# Content trend analysis by year
yearly_contents = df.groupby(['year','type']).size().unstack().fillna(0)

st.header("📅 Yearly Content Trend")
fig1 = px.line(yearly_contents, x=yearly_contents.index, y=yearly_contents.columns,
               title='Yearly Trend of Content Addition',
               labels={'index':'Year', 'value':'No of added contents'},
               color_discrete_sequence=px.colors.qualitative.Set1,
               markers=True)
st.plotly_chart(fig1)

# Sentiment Analysis

def senti(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df = df[df['description'] != 'unknown']
df['sentiment'] = df['description'].apply(senti)
df['label'] = pd.cut(df['sentiment'], bins=[-1, -0.1, 0.1, 1], labels=['Negative', 'Neutral', 'Positive'])
senti_count = (df['label'].value_counts(normalize=True)*100).head(10)

st.header("🎭 Sentiment Analysis")
fig3 = px.pie(values=senti_count.values, names=senti_count.index,
              title="Sentiment Distribution by  Descriptions",
              hole=0.4,
              labels={'label': 'Sentiment', 'value': 'Percentage'},
              color_discrete_sequence=px.colors.sequential.Greens_r
              )

fig3.update_traces(textinfo='percent+label', textfont_size=14)
st.plotly_chart(fig3)


# Top Genres distribution
genr=df['listed_in'].str.split(', ').explode().value_counts().head(10)
st.header("📺 Top Geners")
fig2 = px.bar(x=genr.index, y=genr.values,
              labels={'x':'Genres', 'y':'No of Title'},
              title='Top 1o Geners')
st.plotly_chart(fig2)

# footer
footer()
