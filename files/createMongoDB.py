import collections
import pymongo 
import numpy as np
import pandas as pd
from pymongo import MongoClient
from regex import P


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import pearsonr



def SringMoviesToArray(StrMovies):    
    array = StrMovies.replace("\\", "").replace("'","").replace('"', "").replace("[", "").replace("]","").split(",")
    removeWhiteSpace = []
    removeWhiteSpace.append(array[0])
    for i in range(1,len(array)):
        removeWhiteSpace.append(array[i][1:])
    return removeWhiteSpace

def StringEmbeddingToNumpyArray(strVal):
    return np.fromstring(strVal, dtype='f', sep=' ')


def genre_check(movie_name, df):
    df = df.reset_index()
    genres = df.loc[df['title'] == movie_name]['listed_in'].values
    return str(genres)[2:-2]






def delete_duplicate_data():
    data = collection.find({})
    for doc in data:
        movie1 = doc['movie1']
        movie2 = doc['movie2']
        collection.delete_many({'movie1': movie2 , 'movie2' : movie1 })

    
def sanity_check_print_duplicate_data():
    pairs = collection.find({})
    for pair in pairs:
        movie1 =  pair['movie1']
        movie2 =  pair['movie2']
        obj = collection.find({'movie1': movie2 , 'movie2':movie1})
        for item in obj:
            print(pair)
            print('vs')
            print(item)
    print('done sanity check')

#connect to MongoDB
client = pymongo.MongoClient("mongodb+srv://ramam:ramnight@cluster0.wy9d4.mongodb.net/?retryWrites=true&w=majority")
db = client.netflix
collection = db.movies


def return_two_embeddings(movie_name1, movie_name2, df):
    df = df.reset_index()
    df_embedding1 = str(df.loc[df['title'] == movie_name1]['embedding'].values)[2:-2]
    embedding1 = StringEmbeddingToNumpyArray(df_embedding1)
    df_embedding2 = str(df.loc[df['title'] == movie_name2]['embedding'].values)[2:-2]
    embedding2 = StringEmbeddingToNumpyArray(df_embedding2)
    return embedding1 , embedding2 


def pearson_check(movie_name1, movie_name2, df):  
    embedding1 , embedding2 = return_two_embeddings(movie_name1, movie_name2, df)
    corr, _ = pearsonr(embedding1, embedding2)
    return float(corr)

def cosine_check(movie_name1, movie_name2, df):  
    embedding1 , embedding2 = return_two_embeddings(movie_name1, movie_name2, df)
    corr = cosine_similarity([embedding1], [embedding2])
    return float(corr)

def euclidean_check(movie_name1, movie_name2, df):  
    embedding1 , embedding2 = return_two_embeddings(movie_name1, movie_name2, df)
    corr = euclidean_distances([embedding1], [embedding2])
    return float(corr[0][0])


def create_DB_base_csv(file_name):
    df = pd.read_csv (file_name)
    indexForDB = 0
    for index, row in df.iterrows():
        name = row['title']
        genre = row['listed_in']
        similar_movies = SringMoviesToArray(row['similar_movie'])
        for similar_mov in similar_movies:
            id = name + '$' + similar_mov
            if similar_mov in df['title'].array:
                pearson = pearson_check(name , similar_mov , df)
                cosine = cosine_check(name , similar_mov , df)
                euclidean = euclidean_check(name , similar_mov , df)
                second_movie_genre = genre_check(similar_mov , df)
                netflix = 1
                collection.insert_one({'_id': id, 
                'serial_id': indexForDB,
                'movie1': name,
                'movie2': similar_mov,
                'first_movie_genre': genre, 
                'second_movie_genre': second_movie_genre,
                'pearson': pearson,
                'cosine': cosine,
                'euclidean':euclidean,
                'netflix': netflix  
                })
                indexForDB += 1
                
def create_negative_DB(file_name):
    df = pd.read_csv(file_name)
    indexForDB = 0
    indexHere = 0
    for index, row in df.iterrows():
        name = row['title']
        genre = row['listed_in']
        sample = df.sample()
        sample_name = sample['title'].values[0]
        similar_movies = SringMoviesToArray(row['similar_movie'])
        if sample_name not in similar_movies:
            id = name + '$' + sample_name
            if sample_name in df['title'].array: 
                indexHere += 1
                pearson = pearson_check(name , sample_name , df)
                cosine = cosine_check(name , sample_name , df)
                euclidean = euclidean_check(name , sample_name , df)
                second_movie_genre = genre_check(sample_name , df)
                netflix = 0
                collection.insert_one({'_id': id, 
                'serial_id': indexForDB,
                'movie1': name,
                'movie2': sample_name,
                'first_movie_genre': genre, 
                'second_movie_genre': second_movie_genre,
                'pearson': pearson,
                'cosine': cosine,
                'euclidean':euclidean,
                'netflix': netflix  
                })
                indexForDB += 1






#to use this file u need to:
#create_DB_base_csv
#then delete_duplicate_data
#then u can use sanity check that print all duplicate data (if nothing printed its ok)
#then use create_negative_DB



#print transformers data for report

# name1 = 'Transformers'
# name2 = 'Transformers revenge of the fallen'
# id = name1 + '$' + name2
# embed1 = '-1.035382	0.55451596	0.5010076	-0.20523559	-0.7279057	0.86596453	-0.40977427	-0.21236543	-0.37993565	0.10596954	0.2813178	1.0526767	-0.3635796	0.8141104	0.6068609	0.08688381	1.0870074	-0.3849875	-0.3105793	-0.84923625	0.6776615	-0.40410504	-1.3204294	-0.58526826	-1.196851	-0.119000874	0.12518027	0.11982449	-0.3459172'

# embed2 = '-0.55490464	0.15472439	0.11722883	-0.34559077	-0.11244162	0.030557487	-0.1144814	-0.43434227	0.004961737	-0.12210098	0.5145204	0.12888046	-0.54563165	0.30972135	0.51514804	-0.27728724	0.5988037	-0.2899438	-0.22830224	-0.887797	0.5375573	-0.20658602	-0.29132855	0.12325765	-0.9932721	-0.2904812	-0.4490533	0.10648915	-0.48095635'

# one = np.fromstring(embed1, dtype='f', sep=' ')

# two = np.fromstring(embed2, dtype='f', sep=' ')
# cosine =  cosine_similarity([one], [two])[0][0]
# euclidean = euclidean_distances([one], [two])[0][0]
# pearson= pearsonr(one, two)[0]

# print({'_id': id, 
#                 'movie1': name1,
#                 'movie2': name2,
#                 'first_movie_genre': 'actions, comedy', 
#                 'second_movie_genre': 'actions, comedy',
#                 'pearson': pearson,
#                 'cosine': cosine,
#                 'euclidean':euclidean,
#                 'netflix': 1  
#                 })

