
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#df_keyword = pd.read_csv('KEYWORD.csv', encoding = 'big5')
#df_keyword = df_keyword.drop(['Date'], axis=1)

df_price = pd.read_csv('stg2_FUTURE.csv', encoding = 'big5')
df_price = df_price.drop(['Date'], axis=1)

df_target = df_price.loc[:, ['LME_Nickel_close']]
'''
def keyword_list():
    keyword_data_train = []
    keyword_data_test = []
    for i in range(189):
        a = df_keyword.iloc[i : i+30]
        if i <= 187:
            a = np.array(a.as_matrix())
            keyword_data_train.append(a)
        else:
            a = np.array(a.as_matrix())
            keyword_data_test.append(a)
    return np.array(keyword_data_train), np.array(keyword_data_test)

keyword_train, keyword_test = keyword_list()
'''

def price_list():
    price_data_train = []
    for i in range(54):
        b = df_price.iloc[i+1 : i+11]
        b = np.array(b.as_matrix())
        price_data_train.append(b)
    return np.array(price_data_train)

price_train = price_list()


def target_list():
    target_data_train = []
    for i in range(54):
        c = df_price.iloc[i+11]
        c = np.array(c.as_matrix())
        target_data_train.append(c)
    return np.array(target_data_train)

target_train = target_list()
