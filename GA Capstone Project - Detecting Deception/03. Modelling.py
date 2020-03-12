#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Setting States

#Import relevant standard libraries for processing
import numpy as np
import pandas as pd

#Imports sklearn related libraries
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge,Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict, train_test_split
from sklearn import metrics
from sklearn.preprocessing import Normalizer, StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import LeaveOneOut

from nltk.stem.snowball import SnowballStemmer
from textblob import TextBlob, Word
from string import punctuation
from wordcloud import WordCloud, ImageColorGenerator

import statsmodels.graphics.factorplots

random_state = 42

#Specifiy the current concatenated csv to work from based on that exported in the first Jupyter notebook
working_file = './data/final_data/final_data_27Feb20_144315.csv'


#Read in the working csv file to a pandas dataframe
df_original = pd.read_csv(working_file)
#df_original


# In[192]:


#Investigate head to see if data has been read in correctly. 
df_original.head(5)


# In[193]:


#Lets also have a look at the variables. 
df_original.info()



# In[194]:


#So we have some key references as needed, I will create a list of the gesture variables in our data set
gesture_variables =     ['OtherGestures', 'Smile', 'Laugh', 'Scowl','otherEyebrowMovement', 'Frown', 'Raise', 'OtherEyeMovements', 'Close-R', 'X-Open', 'Close-BE', 'gazeInterlocutor', 'gazeDown', 'gazeUp', 'gazeSide', 'openMouth', 'closeMouth', 'lipsDown', 'lipsUp', 'lipsRetracted', 'lipsProtruded', 'SideTurn', 'downR', 'sideTilt','otherHeadM', 'sideTurnR', 'sideTiltR', 'waggle', 'forwardHead','downRHead', 'singleHand', 'bothHands', 'otherHandM', 'complexHandM', 'sidewaysHand', 'downHands', 'upHands']

#Just in case they are needed I've also recreated the gestures variables for identification of how 
#specific features relate to overarching gesture categories as specified in the MUMIN coding scheme
#(Allwood et al., (2004). The MUMIN multimodal coding scheme. 
#https://www.researchgate.net/publication/228626291_The_MUMIN_multimodal_coding_scheme). 
general_face =          ['Smile', 'Laugh', 'Scowl', 'OtherGestures']
eyebrows =              ['Frown', 'Raise', 'otherEyebrowMovement']
eyes =                  ['X-Open', 'Close-BE', 'Close-R', 'OtherEyeMovements']
gaze =                  ['gazeInterlocutor', 'gazeDown', 'gazeUp', 'gazeSide']
mouth =                 ['openMouth', 'closeMouth']
lips =                  ['lipsDown', 'lipsUp', 'lipsRetracted', 'lipsProtruded', 'target']
head_movements =        ['SideTurn', 'downR', 'sideTilt', 'otherHeadM', 'sideTurnR', 'sideTiltR',
                         'waggle', 'forwardHead', 'downRHead']
hand_gestures =         ['singleHand', 'bothHands', 'otherHandM', 'complexHandM', 'sidewaysHand',
                         'downHands', 'upHands']


# ### Gesture Dimensionality Reduction - PCA
# As we have a lot of gesture features (gestures feature n=36; note also we have lost two columns through during EDA 'otherGaze','backHead' as had no observations/variance) vs. the number target cases (n=109), we have issues of potentially high dimensionality, particularly when we come to fusing models later on. Therefore, I will first investigate whether there are opportunities for dimensionality reduction using PCA.

print("\nPercent of largest class (baseline):", df_original.target_dummy.value_counts(normalize=True).max())

df_unchanged = df_original.copy()

CV_run_times = 10

gesture_CV_scores = {}
vocal_CV_scores = {}
POS_CV_scores = {}
Sentiment_CV_scores = {}
BAG_CV_scores = {}
fusion_class_accuracy = {}
fusion_prob_accuracy = {}

for i, iteration in enumerate(range(1,CV_run_times)):
    df_original, df_original_test = train_test_split(df_unchanged, test_size=0.15)
    
    #Create gesture X and y variables
    X_gesture = df_original[gesture_variables]
    y_gesture = df_original.target_dummy
      
    
    loo=LeaveOneOut()
    
    ### GESTURE SVC MODEL ###
    
    pca_svc = PCA(random_state=random_state)
    svc = SVC(probability=True)
    
    pipe_gest_svc = make_pipeline(pca_svc,svc)
    
    # #####
    params_svc = {'pca__n_components': [10,17,19,25,30],              
                  'svc__C': [.0001,.001,.01,.05, .1,.12,.13,.135,.15,.16,.17,1,3,5,10,20,50,100],              
                  'svc__gamma':np.logspace(-5, 2, 10),              
                  'svc__kernel': ['linear','rbf']}
    
    # # # Set up a gridsearch with above parameters to pass to the pipe pipeline
    grid_pipe_svc = GridSearchCV(pipe_gest_svc,params_svc,verbose=1,n_jobs=-1)
    
    # #8.2. Fit the grid_pipe to the training data
    grid_pipe_svc.fit(X_gesture,y_gesture)
    
    # #Calculate PCA scores for individual cases
    pca_transform_scores_svc = grid_pipe_svc.best_estimator_[0].fit_transform(X_gesture)
    
    # #8.3 And print out the best paramters to the screen
    prediction_class_svc = grid_pipe_svc.predict(X_gesture)
    prediction_class_prob_svc = grid_pipe_svc.predict_proba(X_gesture)
    
    # #Calculate cv scores using leave one out (loo) for on whole data set
    cv_scores_svc = cross_val_score(grid_pipe_svc.best_estimator_, X_gesture, y_gesture, cv=loo)
    
    gesture_CV_scores.update({i:{'cvscores': cv_scores_svc,
                                 'mean_cv': cv_scores_svc.mean(),
                                 'std_cv': cv_scores_svc.std()}})
    
    

    
    
    # ## Vocal Feature Modelling ###
    
    # In[216]:
    
    
    #Again, lets create some lists of features broken into key areas that we can use in modelling if needed.
    #I will create vocal feature counts and time. However, for modelling, I will not include these as they 
    #do not make sense to use given the length of speaking time differs across each instance and, thus, features,
    #will need to be converted to a comparable time metric to facilitate more appropriate comparisons (such rate
    #of speech and ratio of speaking to duration).
    vocal_features_counts = ['Count of Syllables',
                             'Count of filler and pauses']
    vocal_features_time =   ['Speaking time secs w/o pauses',
                             'Duration time secs w pauses']  
    
    #For modelliling purposes, the following vocal features can be used.
    vocal_features_rate =   ['Rate of speech - syb per sec']
    
    vocal_features_ratio =  ['Ratio speaking to duration']
    
    vocal_features_jitter = ['Jitter - local',
                             'Jitter - local, absolute',
                             'Jitter - rap',
                             'Jitter - ppq5',
                             'Jitter - ddp']
    
    vocal_features_FF     = ['Fundamental frequency_f0 Hz', 
                             'SD Fundamental frequency_f0 Hz',
                             'Median Fundamental frequency_f0 Hz',
                             'Minimum Fundamental frequency_f0 Hz',
                             'Maximum Fundamental frequency_f0 Hz',
                             '25th Quantile Fundamental frequency_f0 Hz',
                             '75th Quantile Fundamental frequency_f0 Hz',
                             'diff_max_mix',
                             'diff_25_75']
    
    #All cocal features for X. 
    all_vocal_features =    ['Rate of speech - syb per sec',
                             'Ratio speaking to duration',
                             'Jitter - local',
                             'Jitter - local, absolute',
                             'Jitter - rap',
                             'Jitter - ppq5',
                             'Jitter - ddp',
                             'Fundamental frequency_f0 Hz', 
                             'SD Fundamental frequency_f0 Hz',
                             'Median Fundamental frequency_f0 Hz',
                             'Minimum Fundamental frequency_f0 Hz',
                             'Maximum Fundamental frequency_f0 Hz',
                             '25th Quantile Fundamental frequency_f0 Hz',
                             '75th Quantile Fundamental frequency_f0 Hz',
                             'diff_max_mix',
                             'diff_25_75']
    
    all_vocal_features_alt =    ['Rate of speech - syb per sec',
                             'Ratio speaking to duration',
                             'Jitter - local',
                             'Jitter - local, absolute',
                             'Jitter - rap',
                             'Jitter - ppq5',
                             'Jitter - ddp',
                             'Fundamental frequency_f0 Hz', 
                             'SD Fundamental frequency_f0 Hz',
                             'Median Fundamental frequency_f0 Hz',
                             'Minimum Fundamental frequency_f0 Hz',
                             'Maximum Fundamental frequency_f0 Hz',
                             '25th Quantile Fundamental frequency_f0 Hz',
                             '75th Quantile Fundamental frequency_f0 Hz',
                             'diff_max_mix',
                             'diff_25_75',
                             'Pronunciation_posteriori']
    
    df_original['gender_interaction_diff_25_75'] = df_original.Gender_dummy * df_original.diff_25_75
    
    
    # In[217]:
    
    
    #To ensure that the test features are are changed in the same way, I will apply the same formula 
    df_original_test['gender_interaction_diff_25_75'] = df_original_test.Gender_dummy * df_original_test.diff_25_75
    
    
    # In[218]:
    
    
    #Create gesture X and y variables
    all_vocal_features.append('Gender_dummy')
    all_vocal_features.append('gender_interaction_diff_25_75')
    
    X_vocal = df_original[all_vocal_features]
    y_vocal = df_original.target_dummy
    #And create training and testing class. As we don't have a lot of data for training, 
    #lets keep 10% aside as a validation set. However, we will also investigate the best approach
    
    
    # #### Vocal Modelling - Logistic Regression
    
    # In[219]:
    
    
    #First, lets have a look at the performance for using LeaveOneOut on the entire data-set
    rf_vocal = RandomForestClassifier(random_state=random_state)
    pipe_rf_vocal = make_pipeline(StandardScaler(),rf_vocal)
    
    #####
    params_rf_vocal = {'randomforestclassifier__max_depth':[2,3,4,5],                   
                       'randomforestclassifier__criterion': ['entropy','gini'],                   
                       'randomforestclassifier__max_features': [3,4,5,6]}
    
    # # Set up a gridsearch with above parameters to pass to the pipe pipeline
    grid_rf_vocal = GridSearchCV(pipe_rf_vocal,params_rf_vocal,verbose=1,n_jobs=-1)
    
    # #8.2. Fit the gridsearch to the training data
    grid_rf_vocal.fit(X_vocal,y_vocal)
    
    # #8.3 And print out the best paramters to the screen
    prediction_class_rf_vocal = grid_rf_vocal.predict(X_vocal)
    prediction_class_prob_rf_vocal = grid_rf_vocal.predict_proba(X_vocal)
    
    # #Calculate cv scores using leave one out (loo) for on whole data set
    cv_scores_rf_vocal = cross_val_score(grid_rf_vocal.best_estimator_, X_vocal, y_vocal, cv=loo)
    
    vocal_CV_scores.update({i:{'cvscores': cv_scores_rf_vocal,
                               'mean_cv': cv_scores_rf_vocal.mean(),
                               'std_cv': cv_scores_rf_vocal.std()}})    
    
    # ## Lexical Feature Modelling
    
    # #### Parts of Speech (POS)
    
    # In[231]:
    
    
    #Let's first create new working lexical dataframe to work with for the current modelling. 
    df_lexical = df_original.copy()
    #Drop Gesture variables
    df_lexical.drop(gesture_variables,axis=1,inplace=True)
    #and then all other vocal feature variables and other features not used. 
    df_lexical.drop(['Pronunciation_posteriori', 'Gender', 'Mood_from_mva',
           'Count of Syllables', 'Count of filler and pauses',
           'Rate of speech - syb per sec', 'Duration time secs w pauses',
           'Ratio speaking to duration',
           'Fundamental frequency_f0 Hz', 'SD Fundamental frequency_f0 Hz',
           'Median Fundamental frequency_f0 Hz',
           'Minimum Fundamental frequency_f0 Hz',
           'Maximum Fundamental frequency_f0 Hz',
           '25th Quantile Fundamental frequency_f0 Hz',
           '75th Quantile Fundamental frequency_f0 Hz', 'Jitter - local',
           'Jitter - local, absolute', 'Jitter - rap', 'Jitter - ppq5',
           'Jitter - ddp','pauses_per_minute',
           'log_pauses_per_minute', 'diff_max_mix', 'diff_25_75','target'],axis=1, inplace=True)
    
    
    # In[232]:
    
    
    #Let's first create new working lexical dataframe to work with for the current modelling. 
    df_lexical_test = df_original_test.copy()
    #Drop Gesture variables
    df_lexical_test.drop(gesture_variables,axis=1,inplace=True)
    #and then all other vocal feature variables and other features not used. 
    df_lexical_test.drop(['Pronunciation_posteriori', 'Gender', 'Mood_from_mva',
           'Count of Syllables', 'Count of filler and pauses',
           'Rate of speech - syb per sec', 'Duration time secs w pauses',
           'Ratio speaking to duration',
           'Fundamental frequency_f0 Hz', 'SD Fundamental frequency_f0 Hz',
           'Median Fundamental frequency_f0 Hz',
           'Minimum Fundamental frequency_f0 Hz',
           'Maximum Fundamental frequency_f0 Hz',
           '25th Quantile Fundamental frequency_f0 Hz',
           '75th Quantile Fundamental frequency_f0 Hz', 'Jitter - local',
           'Jitter - local, absolute', 'Jitter - rap', 'Jitter - ppq5',
           'Jitter - ddp','pauses_per_minute',
           'log_pauses_per_minute', 'diff_max_mix', 'diff_25_75','target'],axis=1, inplace=True)
    
    
    # In[233]:
    
    
    X_lexical = df_lexical.transcription.copy()
    
    
    # In[234]:
    
    
    X_lexical_test = df_lexical_test.transcription.copy()
    
    
    # As we are interested in use of parts of speech (e.g. personal pronouns), lets first extract the Parts of Speech (POS) information from the transcriptions. To achieve this, I will use TextBlob, which has a POS module (.tags). 
    
    # In[235]:
    
    
    from collections import Counter
    #Create a dictionary of Parts of Speech counts to append to original lexical dataframe
    pos_dict = {}
    
    for i,x in enumerate(X_lexical):
        textblob_lex = TextBlob(x)
        counts = dict(Counter(tag for word,tag in textblob_lex.tags))
        pos_dict.update({X_lexical.index[i]: counts})
    
    #Create a parts of speech dataframe from the newly created dictionary
    pos_df = pd.DataFrame(pos_dict).T
    
    #And then conctenate the lexical and pos_df dataframes
    df_lexical = pd.concat([df_lexical,pos_df], axis=1)
    
    
    # In[236]:
    
    
    #Create a dictionary of Parts of Speech counts to append to original lexical dataframe
    pos_dict_test = {}
    
    for i,x in enumerate(X_lexical_test):
        textblob_lex = TextBlob(x)
        counts = dict(Counter(tag for word,tag in textblob_lex.tags))
        pos_dict_test.update({df_lexical_test.index[i]: counts})
    
    # #Create a parts of speech dataframe from the newly created dictionary
    pos_df_test = pd.DataFrame(pos_dict_test).T
    
    # #And then conctenate the lexical and pos_df dataframes
    df_lexical_test = pd.concat([df_lexical_test,pos_df_test], axis=1)
    
    # #As some POS are not found in the testing set, lets create these so that issues do not occur later on
    df_lexical_test['FW'] = 0 
    df_lexical_test['NNPS'] = 0 
    
    
    # In[237]:
    
    
    #Now we have our POS counts, lets create a columns list to speed things up when masking if needed
    pos_columns = ['DT', 'NN', 'PRP', 'VBD', 'RB', 'CC',
                   'IN', 'VBN', 'VBG', 'TO', 'WDT', 'VBZ', 'VBP', 'NNS', 'JJ', 'WRB',
                   'NNP', 'PRP$', 'VB', 'RP', 'WP', 'MD', 'PDT', 'CD', 'EX', 'RBR', 'JJR',
                   'UH', 'JJS', 'FW', 'NNPS']
    
    
    # In[238]:
    
    
    #As the length of times speaking differ for each row, lets create a pos rate per minute, which will 
    #Create a ratio variable for rate of part of speech use (POS Per Minute)
    df_lexical = df_lexical.join(df_lexical[pos_columns].apply(lambda x:(df_lexical['Speaking time secs w/o pauses'] /60) * x),rsuffix='_use_per_min')
    
    
    # In[239]:
    
    
    pos_columns = ['DT', 'NN', 'PRP', 'VBD', 'RB', 'CC',
                   'IN', 'VBN', 'VBG', 'TO', 'WDT', 'VBZ', 'VBP', 'NNS', 'JJ', 'WRB',
                   'NNP', 'PRP$', 'VB', 'RP', 'WP', 'MD', 'PDT', 'CD', 'EX',
                   'UH', 'FW', 'NNPS']
    
    
    # In[240]:
    
    
    df_lexical_test = df_lexical_test.join(df_lexical_test[pos_columns].apply(lambda x:(df_lexical_test['Speaking time secs w/o pauses'] /60) * x),rsuffix='_use_per_min')
    
    
    # In[241]:
    
    
    pos_columns_use_per_min = ['DT_use_per_min', 'NN_use_per_min',
           'PRP_use_per_min', 'VBD_use_per_min', 'RB_use_per_min',
           'CC_use_per_min', 'IN_use_per_min', 'VBN_use_per_min',
           'VBG_use_per_min', 'TO_use_per_min', 'WDT_use_per_min',
           'VBZ_use_per_min', 'VBP_use_per_min', 'NNS_use_per_min',
           'JJ_use_per_min', 'WRB_use_per_min', 'NNP_use_per_min',
           'PRP$_use_per_min', 'VB_use_per_min', 'RP_use_per_min',
           'WP_use_per_min', 'MD_use_per_min', 'PDT_use_per_min', 'CD_use_per_min',
           'EX_use_per_min',
           'UH_use_per_min', 'FW_use_per_min',
           'NNPS_use_per_min']
    
    
    # In[242]:
    
    
    #Lets inspect the null values
    df_lexical[pos_columns_use_per_min].isnull().sum()
    
    
    # In[243]:
    
    
    #As we are interested in use, we can replace all null values with 0
    df_lexical[pos_columns_use_per_min] = df_lexical[pos_columns_use_per_min].fillna(0)
    
    
    # In[244]:
    
    
    df_lexical_test[pos_columns_use_per_min].isnull().sum()
    
    
    # In[245]:
    
    
    #As we are interested in use, we can replace all null values with 0
    df_lexical_test[pos_columns_use_per_min] = df_lexical_test[pos_columns_use_per_min].fillna(0)
    
    
    # In[246]:
    
    
    #Lets inspect descriptive statistics of use per minute
    df_lexical[pos_columns_use_per_min].describe().T
    
    
    # In[247]:
    
    
    (df_lexical[pos_columns_use_per_min].describe().T['50%'] == 0).sort_values().index
    
    
    # In[248]:
    
    
    # #As we have a lot of lot counts, lets drop all features that have a median of 0
    # df_lexical.drop(columns =['UH_use_per_min', 'RBR_use_per_min', 'JJR_use_per_min',
    #        'JJS_use_per_min', 'CD_use_per_min', 'FW_use_per_min', 'EX_use_per_min',
    #        'WRB_use_per_min', 'PDT_use_per_min', 'MD_use_per_min',
    #        'WP_use_per_min', 'RP_use_per_min', 'VBZ_use_per_min',
    #        'WDT_use_per_min', 'NNPS_use_per_min'], inplace=True)
    
    pos_columns_use_per_min = ['DT_use_per_min', 'JJ_use_per_min', 'NNS_use_per_min',
           'VBP_use_per_min', 'VB_use_per_min', 'TO_use_per_min',
           'PRP$_use_per_min', 'VBG_use_per_min', 'IN_use_per_min',
           'CC_use_per_min', 'RB_use_per_min', 'VBD_use_per_min',
           'PRP_use_per_min', 'NN_use_per_min', 'VBN_use_per_min',
           'NNP_use_per_min']
    
    
    # In[249]:
    
    
    #Lets have a look at the median POS use per minute broken down by target.
    df_lexical.groupby(by='target_dummy')[pos_columns_use_per_min].agg(np.median).T.sort_values(by=0).plot(kind='barh', figsize=(5,10),rot=0)
    #As can be see, the largest differences between truthful and deceptive statements are on noun use and Prepositions 
    #or subordinating conjunctions with truthful statements more associated with increased use of both
    
    
    # ### POS Modelling
    
    # In[250]:
    
    
    X_POS = df_lexical[pos_columns_use_per_min].copy()
    y_POS = df_lexical.target_dummy.copy()
    
    # ##### POS - SVM
    
    # In[259]:
    
    
    svc_pos = SVC(probability=True)
    
    pipe_pos_svc = make_pipeline(StandardScaler(),svc_pos)
    
    # #####
    params_pos_svc = {'svc__C': [.0001,.001,.01,.05, .1,.12,.13,.135,.15,.16,.17,1,3,5,10,20,40,50,60,70,100],              'svc__gamma':np.logspace(-5, 2, 10),              'svc__kernel': ['linear','rbf']}
    
    # # # Set up a gridsearch with above parameters to pass to the pipe pipeline
    grid_pipe_pos_svc = GridSearchCV(pipe_pos_svc,params_pos_svc,verbose=1,n_jobs=-1)
    
    # #8.2. Fit the grid_pipe to the training data
    grid_pipe_pos_svc.fit(X_POS,y_POS)
    
    # #8.3 And print out the best paramters to the screen
    prediction_class_pos_svc = grid_pipe_pos_svc.predict(X_POS)
    prediction_class_pos_prob_svc = grid_pipe_pos_svc.predict_proba(X_POS)
    
    # #Calculate cv scores using leave one out (loo) for on whole data set
    cv_scores_pos_svc = cross_val_score(grid_pipe_pos_svc.best_estimator_, X_POS,y_POS, cv=loo)
    
    POS_CV_scores.update({i:{'cvscores': cv_scores_pos_svc,
                             'mean_cv': cv_scores_pos_svc.mean(),
                             'std_cv': cv_scores_pos_svc.std()}})    
    
    # ### Sentiment
    
    # As we also interested use of emotion, TextBlob also has a sentiment tool to extract sentiment. Sentiment is broken into polarity and subjectivity. 
    # 
    # * **Polarity:** is float which lies in the range of -1 to 1 where 1 means positive statement and -1 means a negative statement. 
    # * **Subjectivity:** is a measure of the degree to which a sentence contains personal opinion, emotion or judgment whereas objective refers to factual information. Highly levels Subjectivity is also a float which lies in the range of 0 to 1.
    # 
    # To achieve this, I will again use TextBlob, which has a sentiment module (.sentiment). 
    
    # In[262]:
    
    
    #Create a dictionary of Sentiment (polarity and subjectivity) to append to original lexical dataframe
    sentiment_dict = {}
    
    for i,x in enumerate(X_lexical):
        textblob_sent = TextBlob(x)
        sentiment_dict.update({X_lexical.index[i]:{'polarity': textblob_sent.sentiment[0],
                            'subjectivity': textblob_sent.sentiment[1]}})
    
    #Create a sentiment dataframe from the newly created dictionary
    sent_df = pd.DataFrame.from_dict(sentiment_dict, orient='index')
    
    #And then conctenate the lexical and pos_df dataframes
    df_lexical = pd.concat([df_lexical,sent_df], axis=1)
    
    
    # In[263]:
    
    
    #Create a dictionary of Sentiment (polarity and subjectivity) to append to original lexical dataframe
    sentiment_dict_test = {}
    
    for i,x in enumerate(X_lexical_test):
        textblob_sent = TextBlob(x)
        sentiment_dict_test.update({X_lexical_test.index[i]:{'polarity': textblob_sent.sentiment[0],
                            'subjectivity': textblob_sent.sentiment[1]}})
    
    #Create a sentiment dataframe from the newly created dictionary
    sent_df_test = pd.DataFrame.from_dict(sentiment_dict_test, orient='index')
    
    # #And then conctenate the lexical and pos_df dataframes
    df_lexical_test = pd.concat([df_lexical_test,sent_df_test], axis=1)
    
    
    # In[264]:
    
    
    #To make things easier, lets create a list of the sentiment columns to work with. 
    sentiment_columns = ['polarity', 'subjectivity']
    
    
    # In[265]:
    
    
    #Lets have a look at the median sentiment between trutuful and deceptive statments
    df_lexical.groupby(by='target_dummy')[sentiment_columns].agg(np.median).T.sort_values(by=0).plot(kind='bar', figsize=(7,7),rot=0)
    #As can be seen, trutful statements tend to be slightly more positive while deceptive statements tend 
    #to have higher levels of subjectiveity indicating highers levels of personal opinion, emotion, or judgement (vs. factual)
    #These results align to what we may expect in terms of emotional distancing when being deceptive. Specifically, 
    #that those who are deceptive are more likely to use negative terminology (as suggested by lower polarity).
    #We may also expect that a lie that represents something that factually didn't occur, would result in higher
    #levels of subjectivity as shown by higher subjectivity for those who are deceptive. 
    
    
    # ### Sentiment Modelling
    
    # In[266]:
    
    
    X_sentiment = df_lexical[sentiment_columns]
    y_sentiment = df_lexical.target_dummy.copy()
    
    
    
    # #### SVM
    
    # In[270]:
    
    
    svc_sentiment = SVC(probability=True)
    
    pipe_sentiment_svc = make_pipeline(StandardScaler(),svc_sentiment)
    
    # #####
    params_sentiment_svc = {'svc__C': [.0001,.001,.01,.05, .1,.12,.13,.135,.15,.16,.17,1,3,5,10,20,40,50,60,70,100],       
                            'svc__gamma':np.logspace(-5, 2, 10),              
                            'svc__kernel': ['linear','rbf']}
    
    # # # Set up a gridsearch with above parameters to pass to the pipe pipeline
    grid_pipe_sentiment_svc = GridSearchCV(pipe_sentiment_svc,params_sentiment_svc,verbose=1,n_jobs=-1)
    
    # #8.2. Fit the grid_pipe to the training data
    grid_pipe_sentiment_svc.fit(X_sentiment,y_sentiment)
    
    # #8.3 And print out the best paramters to the screen
    prediction_class_sentiment_svc = grid_pipe_sentiment_svc.predict(X_sentiment)
    prediction_class_sentiment_prob_svc = grid_pipe_sentiment_svc.predict_proba(X_sentiment)
    
    # #Calculate cv scores using leave one out (loo) for on whole data set
    cv_scores_sentiment_svc = cross_val_score(grid_pipe_sentiment_svc.best_estimator_, X_sentiment,y_sentiment, cv=loo)
    
    Sentiment_CV_scores.update({i:{'cvscores': cv_scores_sentiment_svc,
                             'mean_cv': cv_scores_sentiment_svc.mean(),
                             'std_cv': cv_scores_sentiment_svc.std()}})  
    
    # ### Bag of Words
    
    # In[276]:
    
    
    X_bow = df_lexical['transcription']
    y_bow = df_lexical.target_dummy
    
    
    # ##### Bag of Words Modelling - Logistic Regression
    
    # In[277]:
    
    
    standard_stop_words = [] #list(list(ENGLISH_STOP_WORDS) + list(punctuation))
    
    bespoke_stop_words = ['t','d']
    
    final_stop_words = standard_stop_words + bespoke_stop_words
    # ##### Bag of Words Modelling - SVM
    
    # In[284]:
    
    
    svc_bag = SVC(probability=True)
    tru_SVD_bag_svc  = TruncatedSVD()
    bag_tfid_svc = TfidfVectorizer(stop_words=final_stop_words)
    
    pipe_bag_svc = make_pipeline(bag_tfid_svc,tru_SVD_bag_svc,svc_bag)
    
    # #####
    params_bag_svc = {'truncatedsvd__n_components': [5,7,20],              
                      'svc__C': [.001,.17,1,3],              
                      'svc__gamma':np.logspace(-5, 2, 10),              
                      'svc__kernel': ['linear','rbf'],
                      'tfidfvectorizer__ngram_range': [(1,1),(1,2)],\
                      'tfidfvectorizer__min_df': [5,7],\
                      'tfidfvectorizer__max_df': [70,90],\
                      'tfidfvectorizer__max_features': [150]}
    
    # # # Set up a gridsearch with above parameters to pass to the pipe pipeline
    grid_pipe_bag_svc = GridSearchCV(pipe_bag_svc,params_bag_svc,verbose=1,n_jobs=-1)
    
    # #8.2. Fit the grid_pipe to the training data
    grid_pipe_bag_svc.fit(X_bow,y_bow)
    
    # #8.3 And print out the best paramters to the screen
    prediction_class_bag_svc = grid_pipe_bag_svc.predict(X_bow)
    prediction_class_bag_prob_svc = grid_pipe_bag_svc.predict_proba(X_bow)
    
    # #Calculate cv scores using leave one out (loo) for on whole data set
    cv_scores_bag_svc = cross_val_score(grid_pipe_bag_svc.best_estimator_, X_bow,y_bow, cv=loo)
    
    BAG_CV_scores.update({i:{'cvscores': cv_scores_bag_svc,
                             'mean_cv': cv_scores_bag_svc.mean(),
                             'std_cv': cv_scores_bag_svc.std()}})     
    
    
    # # Model Fusion
    # ### Prediction Accuracy on Training Set
    
    # In[298]:
        
    # ### Prediction Accuracy on Testing (Holdout) Set
    
    # In[302]:
    
    #Predicting across each model for fusion
    gesture_test_class_predictions = grid_pipe_svc.predict(df_original_test[gesture_variables])
    gesture_test_prob_predictions = grid_pipe_svc.predict_proba(df_original_test[gesture_variables])
    # #Vocal
    vocal_test_class_predictions = grid_rf_vocal.predict(df_original_test[all_vocal_features])
    vocal_test_prob_predictions = grid_rf_vocal.predict_proba(df_original_test[all_vocal_features])
    # #Lexical – POS
    pos_test_class_predictions = grid_pipe_pos_svc.predict(df_lexical_test[pos_columns_use_per_min])
    pos_test_prob_predictions = grid_pipe_pos_svc.predict_proba(df_lexical_test[pos_columns_use_per_min])
    # #Lexical – Sentiment
    sentiment_test_class_predictions = grid_pipe_sentiment_svc.predict(df_lexical_test[sentiment_columns])
    sentiment_test_prob_predictions = grid_pipe_sentiment_svc.predict_proba(df_lexical_test[sentiment_columns])
    # #Lexical – Bag of Words
    bag_test_class_predictions = grid_pipe_bag_svc.predict(df_lexical_test['transcription'])
    bag_test_prob_predictions = grid_pipe_bag_svc.predict_proba(df_lexical_test['transcription'])
    
    

    #Simple classification voting
    fusion_predicted_class = pd.DataFrame(list(zip(df_original_test.target_dummy,
            gesture_test_class_predictions,
            vocal_test_class_predictions,
            pos_test_class_predictions,
            sentiment_test_class_predictions,
            bag_test_class_predictions))).rename(columns={0:'target_true',
                                                        1:'gesture_predicted_class_SVC',
                                                        2:'vocal_predicted_class_rf',
                                                        3:'parts_of_speech_predicted_class_SVC',
                                                        4:'sentiment_predicted_class_SVC',
                                                        5:'bag_predicted_class_SVC'})
    
    fusion_predicted_class['average_class'] = (fusion_predicted_class['gesture_predicted_class_SVC'] +
                                                fusion_predicted_class['vocal_predicted_class_rf'] +
                                                fusion_predicted_class['parts_of_speech_predicted_class_SVC'] +
                                                fusion_predicted_class['sentiment_predicted_class_SVC'] +
                                                fusion_predicted_class['bag_predicted_class_SVC'])/5
    
    fusion_predicted_class['fusion_predicted_class'] = fusion_predicted_class['average_class'].apply(lambda x:1 if x >0.5 else 0)
    
    fusion_class_accuracy_score = accuracy_score(fusion_predicted_class['target_true'],fusion_predicted_class['fusion_predicted_class'])
    
    fusion_class_accuracy.update({i:fusion_class_accuracy_score}) 

    
    # In[313]:
    
    
    #Soft voting classification 
    fusion_predicted_prob = pd.DataFrame(list(zip(df_original_test.target_dummy,
            gesture_test_prob_predictions[:,1],
            vocal_test_prob_predictions[:,1],
            pos_test_prob_predictions[:,1],
            sentiment_test_prob_predictions[:,1],
            bag_test_prob_predictions[:,1]))).rename(columns={0:'target_true',
                                                        1:'gesture_predicted_prob_SVC',
                                                        2:'vocal_predicted_prob_rf',
                                                        3:'parts_of_speech_predicted_prob_SVC',
                                                        4:'sentiment_predicted_prob_SVC',
                                                        5:'bag_predicted_prob_SVC'})
    
    fusion_predicted_prob['average_prob'] = (fusion_predicted_prob['gesture_predicted_prob_SVC'] +
                                                fusion_predicted_prob['vocal_predicted_prob_rf'] +
                                                fusion_predicted_prob['parts_of_speech_predicted_prob_SVC'] +
                                                fusion_predicted_prob['sentiment_predicted_prob_SVC'] +
                                                fusion_predicted_prob['bag_predicted_prob_SVC'])/5
    
    fusion_predicted_prob['fusion_predicted_class'] = fusion_predicted_prob['average_prob'].apply(lambda x:1 if x >0.50 else 0)
    
    fusion_accuracy_score = accuracy_score(fusion_predicted_prob['target_true'],fusion_predicted_prob['fusion_predicted_class'])
    
    fusion_prob_accuracy.update({i:fusion_accuracy_score}) 
    
print(gesture_CV_scores)
print(vocal_CV_scores)
print(POS_CV_scores)
print(Sentiment_CV_scores)
print(BAG_CV_scores)
print(fusion_class_accuracy)
print(fusion_prob_accuracy)