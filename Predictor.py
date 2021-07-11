import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import math




def prepare_Train_Test_Data():  #pomocna metoda koja je pozvana samo jednom da bi podelila podatke
    print("Let's rock it")
    filedsData2 = ['imdb_title_id', 'weighted_average_vote']
    filedsData1 = ['imdb_title_id', 'duration', 'country', 'language', 'director']
    data1 = pd.read_csv("IMDb movies.csv", delimiter=",", usecols=filedsData1)
    print(data1.shape)
    data2 = pd.read_csv("IMDb ratings.csv", delimiter=",", usecols=filedsData2)
    print(data2.shape)
    merged_data = pd.merge(data1, data2, how='left', on=['imdb_title_id', 'imdb_title_id'])
    # s = merged_data.size()
    # print(len(merged_data.columns))
    # print(len(merged_data.row))
    print(merged_data.shape)
    print(merged_data.columns)

    #shuffled = merged_data.sample(frac=1)
    #esult = np.array_split(shuffled, 2) 
    #merged_data = pd.DataFrame(np.random.rand(100, 2)) 
    msk = np.random.rand(len(merged_data)) < 0.8
    train = merged_data[msk]
    test = merged_data[~msk]

    print("================TRAIN==========================")
    print(train)
    print("==========================================")

    print("================TEST==========================")
    print(test)
    print("==========================================")

    train.to_csv('Train.csv')
    test.to_csv('Test.csv')


    #x = merged_data.drop(["weighted_average_vote"], axis=1)
    #y = merged_data.weighted_average_vote
    #print(x.columns)
    #print(y.columns)
    #train_full, test, y_train_full, y_test = train_test_split(x, y,  test_size=0.2) 
    # print("x train size")
    # print("==========================================")
    # print(x_train_full)
    # print("==========================================")

    # print("y train size")
    # print("==========================================")
    # print(y_train_full)
    # print("==========================================")

    # print("x test size")
    # print("==========================================")
    # print(x_test)
    # print("==========================================")

    # print("y test size")
    # print("==========================================")
    # print(y_test)
    # print("==========================================")
    
    

def split_train_validation(data):
    print("==========================================")
    #print(data)
    print("==========================================")

    x = data.drop(["weighted_average_vote"], axis=1)
    y = data.weighted_average_vote
    x_train, x_validation, y_train, y_validation = train_test_split(x, y,  test_size=0.3, shuffle=False) 
    print("x train size")
    print("==========================================")
    #print(x_train)
    print("==========================================")

    print("y train size")
    print("==========================================")
    #print(y_train)
    print("==========================================")

    print("x test size")
    print("==========================================")
    #print(x_validation)
    print("==========================================")

    print("y test size")
    print("==========================================")
    #print(y_validation)
    print("==========================================")
    return  x_train, x_validation, y_train, y_validation


def main(train, test):
    print("Let's rock it!")
    train['weighted_average_vote'] = np.floor(train['weighted_average_vote'].astype(float))
    #train['duration'] = train['duration'].astype(int)
    test['weighted_average_vote'] = np.floor(test['weighted_average_vote'].astype(float))
    # test['duration'] = test['duration'].astype(int)

    labelencoder = LabelEncoder()

    labelencoder.fit(train.weighted_average_vote.astype(int))
    train.weighted_average_vote = labelencoder.transform(train.weighted_average_vote)

    labelencoder.fit(train.duration.astype(int))
    train.duration = labelencoder.transform(train.duration)

    labelencoder.fit(test.weighted_average_vote.astype(int))
    test.weighted_average_vote = labelencoder.transform(test.weighted_average_vote)

    labelencoder.fit(test.duration.astype(int))
    test.duration = labelencoder.fit_transform(test.duration)


    labelencoder.fit(train.imdb_title_id.astype(str))
    train.imdb_title_id = labelencoder.transform(train.imdb_title_id)
    labelencoder.fit(test.imdb_title_id.astype(str))
    test.imdb_title_id = labelencoder.transform(test.imdb_title_id)

    labelencoder.fit(train.country.astype(str))
    train.country = labelencoder.transform(train.country)
    labelencoder.fit(test.country.astype(str))
    test.country = labelencoder.transform(test.country)

    labelencoder.fit(train.language.astype(str))
    train.language = labelencoder.transform(train.language)
    labelencoder.fit(test.language.astype(str))
    test.language = labelencoder.transform(test.language)

    labelencoder.fit(train.director.astype(str))
    train.director = labelencoder.transform(train.director)
    labelencoder.fit(test.director.astype(str))
    test.director = labelencoder.transform(test.director)



    x_train, x_validation, y_train, y_validation = split_train_validation(train)

    # #     <ovo smokoristili za Randomize>
    # # # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # # #n_estimators = [200,220]

    # # # Number of features to consider at every split
    # max_features = ['auto', 'sqrt']

    # # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    # ##max_depth = [10,50, 110]
    # max_depth.append(None)

    # # # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]

    # # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]

    # # # Method of selecting samples for training each tree
    # bootstrap = [True, False]

    # # # Create the random grid
    # random_grid = {'n_estimators': n_estimators,
    #              'max_features': max_features,
    #              'max_depth': max_depth,
    #              'min_samples_split': min_samples_split,
    #              'min_samples_leaf': min_samples_leaf,
    #              'bootstrap': bootstrap}


    # </ovo smokoristili za Randomize>
    ##rf_random = RandomForestClassifier(n_estimators= 200, min_samples_split= 2, min_samples_leaf= 4, max_features= 'sqrt', max_depth= 110, bootstrap= True)
    ##rf_random = RandomForestClassifier(n_estimators= 200, min_samples_split= 2, min_samples_leaf= 4, max_features= sqrt, max_depth= 110, bootstrap= True)
    #rf_random =  RandomForestClassifier(n_estimators = 200, min_samples_split = 5, min_samples_leaf = 4, max_features = 'auto', max_depth = 90, bootstrap= True) #ovaj je konacni
    rf_random =  RandomForestClassifier(n_estimators = 999, min_samples_split = 4, min_samples_leaf = 5, max_features = 'auto', max_depth = 9, bootstrap= True) #ovaj je konacni

    #rf = RandomForestClassifier()
    
    
    #rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 220, cv = 3, verbose=2, random_state=10,n_jobs = -1 )#, n_jobs = -1)


    #ovo je za grid search bilo
    # n_estimators = [999,1000,1001]
    # min_samples_split = [2,3,4]
    # min_samples_leaf = [3, 4, 5]
    # max_features = ['auto']
    # max_depth =  [9,10,11]
    # bootstrap = [True]
    # random_grid = {'n_estimators': n_estimators,
    #               'max_features': max_features,
    #               'max_depth': max_depth,
    #               'min_samples_split': min_samples_split,
    #               'min_samples_leaf': min_samples_leaf,
    #               'bootstrap': bootstrap}

    # rf_random = GridSearchCV(scoring='f1_micro', estimator = rf, param_grid= random_grid, cv = 3, verbose=3, n_jobs = -1)
    #best params: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 110, 'bootstrap': True}

    

    


    print("                                     ")
    print("                                     ")
    print("                                     ")
    #print(x_train)
    #print(y_train)


    # Fit the random search model
    rf_random.fit(x_train,y_train)

    

    prediction = rf_random.predict(x_validation)
    # print(" - - -- - - - - -BEST PARAMS - - - - - - - - - - - - -")
    # print(rf_random.best_params_)
    # print(" - - -- - - - - - - - - - - - - - - - - - -")

    print(" - - -- - - - - -Validation PREDITIONS - - - - - - - - - - - - -")

    print(prediction)
    print(" - - -- - - - - - - Validation score - - - - - - - - - - - -")
    score = f1_score(prediction, y_validation, average='micro')
    print(score)

    # print("       ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~  ~ ~ ~  ~ ~  ~ ~ ~  ~ ~ ~  ~")
    y_test = test['weighted_average_vote']
    # #y_test = np.floor(test['weighted_average_vote'].astype(float))
    x_test = test.drop(['weighted_average_vote'], axis=1) 

    # print(" / / // / /X VALIDATION // / / // / ")
    # print(x_validation)

    # print(" / / // / /X test // / / // / ")
    # print(x_test)

    # print(" / / // / /Y VALIDATION // / / // / ")
    # print(y_validation)

    # print(" / / // / /Y test // / / // / ")
    # print(y_test)

    test_prediction = rf_random.predict(x_test)
    # print(" - - -- - - - - -BEST PARAMS TEST- - - - - - - - - - - - -")
    # print(rf_random.best_params_)
    # print(" - - -- - - - - - - - - - - - - - - - - - -")

    print(" - - -- - - - - -PREDITIONS TEST- - - - - - - - - - - - -")

    print(test_prediction)
    print(" - - -- - - - - - - - TEST SCORE - - - - - - - - - - -")
    test_score = f1_score(test_prediction, y_test, average='micro')
    print(test_score)




if __name__ == "__main__":
    
    ######prepare_Train_Test_Data()
    train = pd.read_csv("Train.csv", delimiter=",")
    print("Train before: ", train.shape)
    test = pd.read_csv("Test.csv", delimiter=",")
    print("Test before: ", test.shape)
    train.dropna(how="any",inplace= True)
    test.dropna(how="any",inplace= True)
 
    print("Train after: ", train.shape)
    print("Test after: ", test.shape)



    main(train,test)
    


    print("DONE!")