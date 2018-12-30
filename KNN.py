import pandas as pd
import numpy as np
class KNN:

    def __init__(self, neigbors):
        neigbors = neigbors

    def train_data(self,train_df,test_df):
        length = len(train_df['sex'])
        train_df['prediction'] = pd.Series(np.random.randn(length), index=train_df.index)
        print(train_df)

# def read_data(filename):
#     data = pd.read_csv(filename, sep="\t")
#     #print(data.columns.values)
#     #data.columns = ["class", "age", "sex", "survived"]
#     data.columns = data.columns.values
#     return data
#with panda

# def train_data(train_df,test_df):
#
#     df_result = pd.DataFrame(0,index = np.arange(len(test_df)),columns=['prediction','knn','id3','naive_bais'])
#     df_result['prediction'] = test_df['survived']
#     print(test_df.columns.size)
#     #val = test_df.columns.size - 1
#     #df_check = test_df.columns[:val]
#     #print(df_check)
#     #df_check = pd.concat(test_df, axis=1, keys=[:val]).reset_index()
#     df_check = test_df.filter(['pclass','age','sex'], axis=1)
#     hamming_distance(train_df,df_check,df_result)



#without panda


# #with panda
# def hamming_distance(train_df,df_check,df_result):
#     idx = 0
#     for i in range(len(df_check)):
#         class_type = df_check['pclass'].iloc[idx]
#         age_type = df_check['age'].iloc[idx]
#         gender_type = df_check['sex'].iloc[idx]
#         idx += 1
#         list = []
#         counter = 0
#         y = len(train_df)
#         for j in range(len(train_df)):
#             t_class_type = train_df['pclass'].iloc[j]
#             t_age_type = train_df['age'].iloc[j]
#             t_gender_type = train_df['sex'].iloc[j]
#             if class_type !=t_class_type:
#                 counter +=1
#             if age_type != t_age_type:
#                 counter +=1
#             if gender_type !=t_gender_type:
#                 counter+=1
#             if counter == 0:
#                 list.append(counter)
#             else:
#                 list.append(counter)
#             counter = 0
#         list.sort()
#         yes = 0
#         no = 0
#         for k in range(5):
#             if list[k] == 0:
#                 yes += 1
#             else:
#                 no +=1
#         if yes > no:
#             df_result.knn[i] = 'yes'
#         else:
#             df_result.knn[i] = 'no'
#     print(df_result)


#train_df = read_data("train.txt")
#test_df = read_data("test.txt")
#train_data(train_df,test_df)
