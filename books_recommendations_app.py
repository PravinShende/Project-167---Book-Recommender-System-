

import re
import operator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from scipy.sparse import csr_matrix
from pandas.api.types import is_numeric_dtype
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import pickle
from PIL import Image

#############################


st.set_page_config(page_title="Books Recommender System App:",layout="wide") #st.beta_set_page_config(layout="wide")

#####################################################################################################


st.markdown('''
<style>
.stApp {
    background-color:#D4F5C9;

} </style>
''', unsafe_allow_html=True)


####### 


st.title(" `Book Recommender System App !`")


image = Image.open('logo book recommender system.jpg')

st.image(image, use_column_width=True)

##############
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")

######################
st.write("""

This app Recommend  the **Required No. of Books** !

""")

st.write("***")  

st.markdown("""
### `Group Members:`
- Mr.Pravin B. Shende
- Mr.Vishal Patil
- Mr.Maniratnam

### `Under Guidance:`

- Mr. Karthik Muskula 
- Mrs. Dhanyapriya Somasundaram

---
""")


# About
expander_bar1 = st.expander("Information Related to The Project ! ")
expander_bar1.markdown("""

## Business Objective:

* **Generate the features from the dataset and use them to recommend the books accordingly to the users.**
## Content

### The Book-Crossing dataset comprises 3 files.
#### Users
Contains the users. Note that user IDs (User-ID) have been anonymized and map to integers. Demographic data is provided (Location, Age) if available. Otherwise, these fields contain NULL-values.
#### Books
Books are identified by their respective ISBN. Invalid ISBNs have already been removed from the dataset. Moreover, some content-based information is given (Book-Title, Book-Author, Year-Of-Publication, Publisher), obtained from Amazon Web Services. Note that in case of several authors, only the first is provided. URLs linking to cover images are also given, appearing in three different flavours (Image-URL-S, Image-URL-M, Image-URL-L), i.e., small, medium, large. These URLs point to the Amazon web site.
#### Ratings
Contains the book rating information. Ratings (Book-Rating) are either explicit, expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by 0.
""")


st.write('---')



st.markdown("""
##### This app helps to recommend the given no. of books using different types of recommendation system types!
* **Python libraries Used:** pandas, streamlit, numpy, matplotlib, seaborn, base64, pickle, re, operator, collections, scipy, sklearn, PIL 
* **Reference Blogs Links** :

    Link 1.[Understanding The Basic Concepts](https://www.analyticsvidhya.com/blog/2021/07/recommendation-system-understanding-the-basic-concepts/)

    Link 2.[A Comprehensive Guide on Recommendation Engines](https://www.analyticsvidhya.com/blog/2022/01/a-comprehensive-guide-on-recommendation-engines-in-2022/)

    Link 3.[Comprehensive Guide to build a Recommendation Engine from scratch ](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/)

    Link 4.[An Easy Introduction to Machine Learning Recommender Systems ](https://www.kdnuggets.com/2019/09/machine-learning-recommender-systems.html)

    Link 5.[Recommender Systems in a Nutshell](https://www.kdnuggets.com/2020/07/recommender-systems-nutshell.html)
* **Reference Youtube Channel Links** :

    Link 1. [Campus X](https://www.youtube.com/watch?v=1YoD0fg3_EM).

    Link 2. [Data Professor](https://www.youtube.com/watch?v=ZZ4B0QUHuNc&list=PLtqF5YXg7GLmCvTswG32NqQypOuYkPRUE).

    Link 3. [Harsh Gupta](https://www.youtube.com/watch?v=UN4DaSAZel4&list=PLuU3eVwK0I9PT48ZBYAHdKPFazhXg76h5).
""")

st.write('---')

##############

img=Image.open("sidebar_logo_header.jpg")
st.sidebar.image(img,use_column_width=True)

###################
st.sidebar.header('Link to Original Csv Files')

##################


st.sidebar.markdown("""
  [Users CSV input file](https://github.com/PravinShende/datasets_book_recommender_system/blob/main/Users.csv)
 """)

st.sidebar.markdown("""
  [Rating CSV input file](https://github.com/PravinShende/datasets_book_recommender_system/blob/main/Ratings.csv)
 """)

st.sidebar.write('***')

st.sidebar.header('User Input Features')

#################################
Books= st.sidebar.file_uploader("Upload your input books csv file", type=["csv"])
Users= st.sidebar.file_uploader("Upload your input Users CSV file", type=["csv"])
Ratings= st.sidebar.file_uploader("Upload your input Ratings CSV file", type=["csv"])

st.sidebar.write('---')


if Books is not None:
    books = pd.read_csv(Books)
else:
    st.info('Awaiting for Books CSV file to be uploaded.')


if Users is not None:
    users = pd.read_csv(Users)
else:
    st.info('Awaiting for Users CSV file to be uploaded.')


if Ratings is not None:
    ratings = pd.read_csv(Ratings)
else:
    st.info('Awaiting for Ratings CSV file to be uploaded.')


#########################
#number=st.sidebar.slider('No.Of Books to Recommend ',1,20)

st.sidebar.header('Select The No. of Books to Recommnend')
#number = st.sidebar.selectbox('No. of Books', list(range(1,31)))
#number = st.sidebar.slider('Select The No. of Books to Recommnend', 1,30,8)
number = st.sidebar.number_input('Value should be between 2 and 20 ', 2,20,4)

st.sidebar.write("***")



#####################

# books = pd.read_csv("Books.csv")
# users = pd.read_csv("Users.csv")
# ratings = pd.read_csv("Ratings.csv")

###########

st.subheader("Below Tab will show you the original Dataframes of all 3 Files !")


######

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
########

tab1, tab2,tab3 = st.tabs(['Books Dataframe','Users Dataframe','Ratings Dataframe'])

tab1.subheader("Original Books Dataframe: ")
tab1.write(books.head(5))

tab2.subheader('Original Users Dataframe:')
tab2.write(users.head(5))

tab3.subheader("Original Ratings Dataframe:")
tab3.write(ratings.head(5))

st.write("***")

expander_bar3 = st.expander("Information Related to Dimensions of all 3 Dataframes ! ")

expander_bar3.write('Data Dimension of Books Dataset: ' + str(books.shape[0]) + ' rows and ' + str(books.shape[1]) + ' columns.')
expander_bar3.write('Data Dimension of Users Dataset: ' + str(users.shape[0]) + ' rows and ' + str(users.shape[1]) + ' columns.')
expander_bar3.write('Data Dimension of Ratings Dataset: ' + str(ratings.shape[0]) + ' rows and ' + str(ratings.shape[1]) + ' columns.')

################
books_p=pd.read_csv("books_csv_p.csv")

users_p=pd.read_csv("users_csv_p.csv")

ratings_p=pd.read_csv("ratings_csv_p.xlsx")

st.write("***")
#######
st.subheader("Below Tab will show you the Processed Dataframes of all 3 Files !")

tab4, tab5,tab6 = st.tabs(['Books Dataframe','Users Dataframe','Ratings Dataframe'])

tab4.subheader("Processed Books Dataframe: ")
tab4.write(books_p.head(5))

tab5.subheader('Processed Users Dataframe:')
tab5.write(users_p.head(5))

tab6.subheader("Processed Ratings Dataframe:")
tab6.write(ratings_p.head(5))

st.write("***")

dataset_p=pd.read_csv("dataset_p.csv")

############

list_books= ["Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))","Harry Potter and the Goblet of Fire (Book 4)",'Blow Fly: A Scarpetta Novel','The Testament','The Sneetches and Other Stories','The Two Towers (The Lord of the Rings, Part 2)','Postmarked Yesteryear: 30 Rare Holiday Postcards','The Da Vinci Code','Wild Animus']

#list_books= list(dataset_p['Book-Title'])

#random_book_no= randrange(0, len(list_books) )

st.sidebar.subheader('Randomly Selected Books or Can give any book Name:')
# bookName = st.sidebar.selectbox('Random Book',
#                                         list_books,
#                                         index = random_book_no)

bookName = st.sidebar.selectbox('Random Book',
                                        list_books)
                                        

#bookName= "The Kitchen God's Wife"


#####################
st.write('***')

if st.button('Click to See Merged Dataframe'):
    st.header('This is the Merged Dataframe of all 3 Processed Dataframes!')
    st.write(dataset_p.head())



# # Sidebar - Book  selection
# sorted_books = sorted(dataset_p['Book-Title'].unique())

# bookName = st.sidebar.selectbox('Select Any One Book ', list(sorted_books))

#################



dataset1_p=pd.read_csv("dataset1_p.csv")
dataset2_p=pd.read_csv("dataset2_p.csv")

###################

list_countries=["canada","india","usa","germany","spain","italy","france","portugal","austrailia","russia"]

#list_countries= list(dataset1_p['Country'])

#random_country_no= randrange(0, len(list_countries) )

st.sidebar.subheader('Randomly Selected Country or Can give any city,state or Country Name:')
# place= st.sidebar.selectbox('Random Country',
#                                         list_countries,
#                                         index = random_country_no)

place= st.sidebar.selectbox('Random Country',
                                        list_countries)


#place="germany"


######################

st.write("***")

st.subheader("Below Tabs will show you the Data Visulization !")

tab1_d, tab2_d,tab3_d,tab4_d, tab5_d,tab6_d,tab7_d, tab8_d,tab9_d,tab10_d, tab11_d = st.tabs(['books published yearly','books by an author','books published by a publisher','Frequency acc. to the Rating','Frequency acc. to the Explicit Rating','Age Distribution','Citiwise No. of Readers','Statewise No. of Readers','Countriwise No. of Readers','USA top states  No. of Readers','Top Rated books'])


###################
publications = {}
for year in books_p['Year-Of-Publication']:
    if str(year) not in publications:
        publications[str(year)] = 0
    publications[str(year)] +=1

publications = {k:v for k, v in sorted(publications.items())}

fig1 = plt.figure(figsize =(55, 15))
plt.bar(list(publications.keys()),list(publications.values()), color = 'blue')
plt.ylabel("Number of books published")
plt.xlabel("Year of Publication")
plt.title("Number of books published yearly")
plt.margins(x = 0)
plt.show()
tab1_d.subheader("Number of books published yearly")
tab1_d.pyplot(fig1)



#########################

fig2=plt.figure(figsize=(15,6))
sns.countplot(y="Book-Author", data=books_p,order=books_p['Book-Author'].value_counts().index[0:15])
plt.title("No of books by an author (Top 15)")
plt.show()
tab2_d.subheader("No of books written  by an author (Top 15)")
tab2_d.pyplot(fig2)

#########################

fig3=plt.figure(figsize=(15,6))
sns.countplot(y="Publisher", data=books_p,order=books_p['Publisher'].value_counts().index[0:15])
plt.title("No of books published by a publisher (Top 15)")
plt.show()
tab3_d.subheader("No of books published by a publisher (Top 15)")
tab3_d.pyplot(fig3)

##########################

fig4=plt.figure(figsize=(8,4))
sns.countplot(x="Book-Rating", data=ratings_p)
plt.title("Frequency acc. to the Rating")
plt.show()
tab4_d.subheader("Frequency Distribution acc. to Rating ")
tab4_d.pyplot(fig4)

#########################

fig5=plt.figure(figsize=(8,4))
data = ratings_p[ratings_p['Book-Rating'] != 0]
sns.countplot(x="Book-Rating", data=data)
plt.title("Explicit Ratings")
plt.show()
tab5_d.subheader("Frequency Distribution acc. to Explicit Rating ")
tab5_d.pyplot(fig5)

############################

fig6=plt.figure(figsize=(8,4))
users_p.Age.hist(bins=[10*i for i in range(1, 10)])     
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
tab6_d.subheader("Readers Age distribution ")
tab6_d.pyplot(fig6)

#############################

fig7=plt.figure(figsize=(20,6))
sns.countplot(x="City", data=users_p,order=users_p['City'].value_counts().index[0:15])
plt.title("No of readers from each city (Top 15)")
plt.show()
tab7_d.subheader("top 15 cities having highest no. of readers ")
tab7_d.pyplot(fig7)

############################

fig8=plt.figure(figsize=(20,6))
sns.countplot(x="State", data=users_p,order=users_p['State'].value_counts().index[0:15])
plt.title("No of readers from each state (Top 15)")
plt.show()
tab8_d.subheader("top 15 states having highest no. of readers ")
tab8_d.pyplot(fig8)

######################

fig9=plt.figure(figsize=(15,8))
sns.countplot(y="Country", data=users_p, order=users_p['Country'].value_counts().index[0:10])
plt.title("No of readers from each country (Top 10)")
plt.show()
tab9_d.subheader("top 15 country having highest no. of readers ")
tab9_d.pyplot(fig9)

######################

data=users_p[users_p['Country']=='usa']
fig10=plt.figure(figsize=(20,6))
sns.countplot(x="State", data=data,order=data['State'].value_counts().index[0:15])
plt.title("No of readers from states of USA (Top 15)")
plt.show()
tab10_d.subheader("top 15 states of USA having highest No. of readers ")
tab10_d.pyplot(fig10)

#################################

fig11=plt.figure(figsize=(15,8))
sns.countplot(y="Book-Title", data=dataset_p, order=dataset_p['Book-Title'].value_counts().index[0:15])
plt.title("Number of Ratings for a book (Top 15)")
plt.show()
tab11_d.subheader("top 15 books having highest ratings ")
tab11_d.pyplot(fig11)

###############################################################################################################


st.header("Recommendation System: ")


# bookName = input("Enter a book name: ")
# number = int(input("Enter number of books to recommend: "))

# Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))


st.subheader("1. Popularity Based (Top In whole collection)")

def popularity_based(dataframe, n):
    if n >= 1 and n <= len(dataframe):
        data = pd.DataFrame(dataframe.groupby('ISBN')['Book-Rating'].count()).sort_values('Book-Rating', ascending=False).head(n)
        result = pd.merge(data, books_p, on='ISBN')
        return result
    return "Invalid number of books entered!!"


st.write("Top", number, "Popular books are: ")
st.write(popularity_based(dataset1_p, number))

st.write("***")


################################################################################

st.subheader("2. Popularity Based (Top In a given place)")

def search_unique_places(dataframe, place):
    place = place.lower()

    if place in list(dataframe['Country'].unique()):
        return dataframe[dataframe['Country'] == place]
    
    elif place in list(dataframe['State'].unique()):
        return dataframe[dataframe['State'] == place]

    elif place in list(dataframe['City'].unique()):
        return dataframe[dataframe['City'] == place]
    
    else:
        return "Invalid Entry"



#place = input("Enter the name of place: ")
data = search_unique_places(dataset1_p, place)

if isinstance(data, pd.DataFrame):
    data = popularity_based(data, number)


st.write("Top", number, "Popular books  Beased on country are : ")
st.write(data)

st.write("***")

##################################################################

st.subheader("3. Books by same author, publisher of given book name")


def printBook(k, n):
    z = k['Book-Title'].unique()
    for x in range(len(z)):
        st.write(z[x])
        if x >= n-1:
            break



def get_books(dataframe, name, n):

    #global k2   # This line I have added bcz of error generation!

    st.subheader("\nBooks by same Author:\n")
    au = dataframe['Book-Author'].unique()

    data = dataset1_p[dataset1_p['Book-Title'] != name]

    if au[0] in list(data['Book-Author'].unique()):
        k2 = data[data['Book-Author'] == au[0]]
    k2 = k2.sort_values(by=['Book-Rating'])
    printBook(k2, n)

    st.subheader("\n\nBooks by same Publisher:\n")
    au = dataframe['Publisher'].unique()

    if au[0] in list(data['Publisher'].unique()):
        k2 = pd.DataFrame(data[data['Publisher'] == au[0]])
    k2=k2.sort_values(by=['Book-Rating']) 
    printBook(k2, n)




if bookName in list(dataset1_p['Book-Title'].unique()):
    d = dataset1_p[dataset1_p['Book-Title'] == bookName]
    get_books(d, bookName, number)
else:
    st.write("Invalid Book Name!!")

st.write("***")

################################################################################


st.subheader("4. Books popular Yearly")

data = pd.DataFrame(dataset1_p.groupby('ISBN')['Book-Rating'].count()).sort_values('Book-Rating', ascending=False)
data = pd.merge(data, books_p, on='ISBN')

years = set()
indices = []
for ind, row in data.iterrows():
    if row['Year-Of-Publication'] in years:
        indices.append(ind)
    else:
        years.add(row['Year-Of-Publication'])

data = data.drop(indices)
data = data.drop('Book-Rating', axis = 1)
data = data.sort_values('Year-Of-Publication')

#pd.set_option("display.max_rows", None, "display.max_columns", None)
st.write(data)

st.write("***")

#######################################################################################

st.subheader("5. Average Weighted Ratings")


df = pd.read_pickle('weightedData')

## C - Mean vote across the whole
C = df['Average Rating'].mean()

## Minimum number of votes required to be in the chart
m = df['Total-Ratings'].quantile(0.90)


def weighted_rating(x, m=m, C=C): 
    v = x['Total-Ratings']    #v - number of votes
    R = x['Average Rating']   #R - Average Rating   
    return (v/(v+m) * R) + (m/(m+v) * C)



df = df.loc[df['Total-Ratings'] >= m]

df['score'] = df.apply(weighted_rating, axis=1)


df = df.sort_values('score', ascending=False)

st.subheader("Recommended Books:-\n")
st.write(df.head(number))

st.write("***")

########################################################################################

st.subheader("6. Collaborative Filtering (User-Item Filtering)")

# Selecting books with total ratings equals to or more than 50 (Because of availability of limited resources)

df = pd.DataFrame(dataset1_p['Book-Title'].value_counts())
df['Total-Ratings'] = df['Book-Title']
df['Book-Title'] = df.index
df.reset_index(level=0, inplace=True)
df = df.drop('index',axis=1)

df = dataset1_p.merge(df, left_on = 'Book-Title', right_on = 'Book-Title', how = 'left')
df = df.drop(['Year-Of-Publication','Publisher','Age','City','State','Country'], axis=1)

popularity_threshold = 50
popular_book = df[df['Total-Ratings'] >= popularity_threshold]
popular_book = popular_book.reset_index(drop = True)


# User - Item Collaborative Filtering


testdf = pd.DataFrame()
testdf['ISBN'] = popular_book['ISBN']
testdf['Book-Rating'] = popular_book['Book-Rating']
testdf['User-ID'] = popular_book['User-ID']
testdf = testdf[['User-ID','Book-Rating']].groupby(testdf['ISBN'])



listOfDictonaries=[]
indexMap = {}
reverseIndexMap = {}
ptr=0

for groupKey in testdf.groups.keys():
    tempDict={}
    groupDF = testdf.get_group(groupKey)
    for i in range(0,len(groupDF)):
        tempDict[groupDF.iloc[i,0]] = groupDF.iloc[i,1]
    indexMap[ptr]=groupKey
    reverseIndexMap[groupKey] = ptr
    ptr=ptr+1
    listOfDictonaries.append(tempDict)

dictVectorizer = DictVectorizer(sparse=True)
vector = dictVectorizer.fit_transform(listOfDictonaries)
pairwiseSimilarity = cosine_similarity(vector)



def printBookDetails(bookID):
    st.write(dataset1_p[dataset1_p['ISBN']==bookID]['Book-Title'].values[0])
    # """
    # print("Title:", dataset1[dataset1['ISBN']==bookID]['Book-Title'].values[0])
    # print("Author:",dataset1[dataset['ISBN']==bookID]['Book-Author'].values[0])
    # #print("Printing Book-ID:",bookID)
    # print("\n")
    # """

def getTopRecommandations(bookID):
    collaborative = []
    row = reverseIndexMap[bookID]
    st.subheader("Input Book:")
    printBookDetails(bookID)
    
    st.subheader("\nRECOMMENDATIONS:\n")
    
    mn = 0
    similar = []
    for i in np.argsort(pairwiseSimilarity[row])[:-2][::-1]:
          if dataset1_p[dataset1_p['ISBN']==indexMap[i]]['Book-Title'].values[0] not in similar:
                if mn>=number:
                      break
                mn+=1
                similar.append(dataset1_p[dataset1_p['ISBN']==indexMap[i]]['Book-Title'].values[0])
                printBookDetails(indexMap[i])
                collaborative.append(dataset1_p[dataset1_p['ISBN']==indexMap[i]]['Book-Title'].values[0])
    return collaborative



k = list(dataset1_p['Book-Title'])
m = list(dataset1_p['ISBN'])

collaborative = getTopRecommandations(m[k.index(bookName)])

st.write("***")



##############################################################################

st.subheader("7. Correlation Based")

popularity_threshold = 50

user_count = dataset1_p['User-ID'].value_counts()
data = dataset1_p[dataset1_p['User-ID'].isin(user_count[user_count >= popularity_threshold].index)]
rat_count = data['Book-Rating'].value_counts()
data = data[data['Book-Rating'].isin(rat_count[rat_count >= popularity_threshold].index)]

matrix = data.pivot_table(index='User-ID', columns='ISBN', values = 'Book-Rating').fillna(0)




average_rating = pd.DataFrame(dataset1_p.groupby('ISBN')['Book-Rating'].mean())
average_rating['ratingCount'] = pd.DataFrame(ratings_p.groupby('ISBN')['Book-Rating'].count())
average_rating.sort_values('ratingCount', ascending=False)


isbn = books_p.loc[books_p['Book-Title'] == bookName].reset_index(drop = True).iloc[0]['ISBN']
row = matrix[isbn]
correlation = pd.DataFrame(matrix.corrwith(row), columns = ['Pearson Corr'])
corr = correlation.join(average_rating['ratingCount'])

res = corr.sort_values('Pearson Corr', ascending=False).head(number+1)[1:].index
corr_books = pd.merge(pd.DataFrame(res, columns = ['ISBN']), books_p, on='ISBN')
st.subheader("\n Recommended Books: \n")
st.write(corr_books)

st.write("***")

##############################################################################################


st.subheader("8. Nearest Neighbours Based")

popularity_threshold = 50


data = (dataset1_p.groupby(by = ['Book-Title'])['Book-Rating'].count().reset_index().
        rename(columns = {'Book-Rating': 'Total-Rating'})[['Book-Title', 'Total-Rating']])

result = pd.merge(data, dataset1_p, on='Book-Title')  # this code i hv removed (, left_index = True)
result = result[result['Total-Rating'] >= popularity_threshold]
result = result.reset_index(drop = True)

matrix = result.pivot_table(index = 'Book-Title', columns = 'User-ID', values = 'Book-Rating').fillna(0)
up_matrix = csr_matrix(matrix)




model = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model.fit(up_matrix)

distances, indices = model.kneighbors(matrix.loc[bookName].values.reshape(1, -1), n_neighbors = number+1)
st.subheader("\nRecommended books:\n")
for i in range(0, len(distances.flatten())):
    if i > 0:
        st.write(matrix.index[indices.flatten()[i]]) 


st.write("***")

##############################################################################


