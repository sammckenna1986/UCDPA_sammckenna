

import pandas as pd
import numpy as np
import random
import re


def import_and_explore_data():
    data = pd.read_csv('ramen-ratings.csv')
    df = pd.DataFrame(data)
    print(df.head())
    print(df.describe())

    # https://docs.python.org/3/library/re.html
    # https://www.w3schools.com/python/python_regex.asp

def regex():
    data = pd.read_csv('ramen-ratings.csv')
    df = pd.DataFrame(data)
    variety = df['Variety']
    country = df['Country']
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    for line in re.findall(".*Chicken.*", str(variety)):
        if re.findall("Grilled",line):
            #for line in re.split('Chicken', line, maxsplit=2):
            for line in re.findall('.*Flavor$',line):
                print ('The following varieties include the words  "Chicken" and "Grilled" and ends with the word "Flavor: ' +  line)

    for line in re.findall(".*Pork.*", (str(country + ' ' + variety))):
        if re.findall(".*Artificial.*", line):
            if re.findall(".*Vietnam.*", line):
                print('This row contains rows where regex found the words "Pork", then "Artificial", then "Vietnam": ',line)

# replace missing values or drop duplicates
def delete_random_values_then_drop_na():
    data = pd.read_csv('ramen-ratings.csv')
    df_unclean = pd.DataFrame(data)
    df_clean = pd.DataFrame(data)
    # https://stackoverflow.com/questions/61017329/how-to-randomly-delete-10-attributes-values-from-df-in-pandas
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html -- used these to just delete a random sample of cells of the data set.
    for col in df_unclean.columns:
        df_unclean.loc[df_unclean.sample(frac=0.15).index,col] = np.nan # changed it to 15%
        df_unclean_drop_row = df_unclean.drop(['Top Ten'],axis=1) # dropped the 'Top Ten' column as it was mostly NAN out of 2580 columns
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html#pandas-dataframe-drop
        # Then going to drop the rows that have NAN -- https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
        df_unclean_drop_row_then_DROPNA = df_unclean_drop_row.dropna(axis=0)

    #print(df_unclean_drop_row_then_DROPNA) #printing the still 965 rows left without any NAN
    print('How many rows have NAN left in them after I randomly deleted values and then removed the rows with NAN in them: \n',df_unclean_drop_row_then_DROPNA.isnull().sum() / len(df_unclean_drop_row_then_DROPNA)) # this just shows how many of the columns have NAN in them.


def delete_random_values_inStars_then_fill():
    data = pd.read_csv('ramen-ratings.csv')
    df_unclean = pd.DataFrame(data)
    stars_data = df_unclean[['Stars','Review']]
    # https://stackoverflow.com/questions/61017329/how-to-randomly-delete-10-attributes-values-from-df-in-pandas
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html -- used these to just delete a random sample of cells of the data set.
    stars_data['Stars'].loc[stars_data.sample(frac=0.10).index] = np.nan # changed it to 10%
    stars_new = pd.DataFrame(stars_data)
    ratings = stars_new.rename(columns={'Stars': 'Rating'})

    deleted_stars_data = df_unclean.drop(['Stars','Top Ten'], axis=1)
    merge_ratings_deleted_stars = pd.merge(deleted_stars_data, ratings,on='Review')

    #pd.set_option("display.max_rows", None,"display.max_columns", None)
    merge_ratings_deleted_stars.fillna(value=3,inplace=True) #https://www.w3schools.com/PYTHON/pandas/ref_df_fillna.asp
    merge_ratings_deleted_stars.sort_values('Review',axis=0,inplace=True)
    merge_ratings_deleted_stars.to_csv('Ramen_NEW.csv',index=False)

    # choosing a random 10 ramen meals to make out of the data base and then putting them into a list to try
    #https://stackoverflow.com/questions/10125568/how-to-randomly-choose-multiple-keys-and-its-value-in-a-dictionary-python
    list_of_ramen_meals = random.choices(np.array(merge_ratings_deleted_stars),k=10)
    saved_list_of_ramen_meals = list_of_ramen_meals

    ramen_meals_back_to_df = pd.DataFrame(saved_list_of_ramen_meals, columns=['Review','Brand','Variety','Style','Country','Rating'])
    ramen_meals_back_to_df.to_csv('first_10_Ramen_meals.csv', index=False)

    print('This is the first one of the radomly chosen 10 ramen meals from the entire repository - "saved list of ramen meals"', saved_list_of_ramen_meals[0])
    print('The following is going to be your first ramen meal, enjoy! : ' + str(saved_list_of_ramen_meals[0][2]))


    #print(stars_new.isnull().sum() / len(stars_new)) # this just shows how many of the columns have NAN in them.



regex()
delete_random_values_then_drop_na()
delete_random_values_inStars_then_fill()

