import csv

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from surprise import SVD, accuracy
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate

def ContentBased(name, artist, df):
    try:
        songData = df[(df['track_name'] == name) & (df["track_artist"] == artist)]
        similar = df.copy()

        properties = similar.loc[:,['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

        similar['Similarity Score'] = cosine_similarity(properties, properties.to_numpy()[songData.index[0],None]).squeeze()
        similar.rename(columns={'track_name': f'Songs Similar to {name}'}, inplace=True)
        similar = similar.sort_values(by= 'Similarity Score', ascending=False)
        similar = similar[['track_artist', f'Songs Similar to {name}', 'track_album_release_date', 'track_popularity']]
        similar.reset_index(drop=True, inplace=True)

        return similar.iloc[1:11]
    except:
        print("Song not in dataset")
# Code below is for content based
#songsDF = pd.read_csv("spotify_songs.csv")
#songsDF = songsDF.drop_duplicates(subset=['track_name'])

#recommendation = ContentBased("Torn - KREAM Remix", "Ava Max", songsDF)

#print(recommendation.iloc[0])
#display(recommendation)
#recommendation.to_csv('out.csv', index = False)

# --------------------------------------------------------------------------

# For Collaborative filtering
# First read in the data
#songInfoDF = pd.read_csv("datasets/collabFilter/song_data.csv")
#userHistoryDF = pd.read_table("datasets/collabFilter/10000.txt", header=None)
#userHistoryDF.columns = ['user_id', 'song_id', 'listen_count']

# Combine both dataframes to map songs with user_id's and listen_counts
# both dataframes have song_id key so join on that.
#df = pd.merge(userHistoryDF, songInfoDF.drop_duplicates(['song_id']), on="song_id", how="left")

#Save the df to a CSV
#df.to_csv('listenDataMerged.csv', quoting=csv.QUOTE_NONNUMERIC)

#ALL ABOVE CODE IS DONE WE HAVE THE CSV WITH LISTENS AND SONGS.
songDF = pd.read_csv("listenDataMerged.csv")
songDF.drop(songDF.columns[songDF.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

# Grab songs that are listened to by the users
songCount = songDF.groupby('song_id')['user_id'].count()
#sieve out songs that have been listened to more than n times
songIDs = songCount[songCount > 150].index.to_list()

#grab users that have listened to more than n songs
userCount = songDF.groupby('user_id')['song_id'].count()
userIDs = userCount[userCount > 15].index.to_list()

#Apply filters to DF into new DF
songFilteredDF = songDF[(songDF['user_id'].isin(userIDs)) & (songDF['song_id'].isin(songIDs))].reset_index(drop=True)

#use binning technique
bin = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2214]
songFilteredDF['listen_count'] = pd.cut(songFilteredDF['listen_count'], bins=bin, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

listenCount = pd.DataFrame(songFilteredDF.groupby('listen_count').size(), columns=['count']).reset_index(drop=False)

scale = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(songFilteredDF[['user_id', 'song_id', 'listen_count']], scale)

# Splitting for training data and test data
trainData, testData = train_test_split(data, test_size=.25)

#Definning parameters for model
params = {
    'n_factors': [160],
    'n_epochs': [100],
    'lr_all': [0.001, 0.005],
    'reg_all': [0.08, 0.1]
}

#gsvd = GridSearchCV(SVD, params, measures=['rmse'], cv=3, joblib_verbose=4, n_jobs=-2)
#gsvd.fit(data)
#algo = gsvd.best_estimator['rmse']
#print(gsvd.best_score['rmse'])
#print(gsvd.best_params['rmse'])
#best parameters are
#n_factors = 160, n_epochs = 100, lr_all = 0.001, reg_all = 0.1

#cross_validate(algo, data, measures=['rmse'], cv=5, verbose=True)

model = SVD(n_factors=160, n_epochs=100, lr_all=0.001, reg_all=0.1)
model.fit(trainData)
predictions = model.test(testData)
#print(predictions)
print(f"The RMSE is {accuracy.rmse(predictions, verbose=True)}")

#Here we input a userID, and then a song in the dataset, predict the rating the user would
print(model.predict('bc5bd05ea8cab961847ece232725178e86503638', 'SOAUWYT12A81C206F1'))

#Sources used:
# https://medium.com/@jonahflateman/using-surprise-in-python-with-a-recommender-system-2d6030140926
# https://surprise.readthedocs.io/en/stable/prediction_algorithms.html
# https://github.com/ugis22/music_recommender/blob/master/collaborative_recommender_system/CF_matrix_fact_music_recommender.ipynb


