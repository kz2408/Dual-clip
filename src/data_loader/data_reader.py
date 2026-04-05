# import pandas as pd

# df1 = pd.read_csv('/data2/yanjiexuan/voc/files/VOC2007/classification_trainval.csv')
# df2 = pd.read_csv('/data2/yanjiexuan/voc/files/VOC2007/classification_test.csv')
# # print(df.head())

# df1.replace(-1, 0, inplace=True)
# df2.replace(-1, 0, inplace=True)

# object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
#                      'bottle', 'bus', 'car', 'cat', 'chair',
#                      'cow', 'diningtable', 'dog', 'horse',
#                      'motorbike', 'person', 'pottedplant',
#                      'sheep', 'sofa', 'train', 'tvmonitor']

# base_categories = ['aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'chair','cow', 'diningtable', 'horse','motorbike', 'person', 'train']
novel_categories = ['dog', 'sofa', 'cat', 'pottedplant', 'tvmonitor','sheep']


# for i in base_categories:
#     df1.drop(df1[df1[i] == 1].index,inplace = True)
#     df2.drop(df2[df2[i] == 1].index,inplace = True)
# # print(df1.head())

# for i in base_categories:
#     df1.drop(columns = [i], inplace = True)
#     df2.drop(columns = [i], inplace = True)

# # train_base
# # test_base
# df1.to_csv('/home/yanjiexuan/multi-label-fsl/RC-Tran/idx/voc/classification_Train_novel.csv',index = False)
# df2.to_csv('/home/yanjiexuan/multi-label-fsl/RC-Tran/idx/voc/classification_Test_novel.csv',index = False)
# # print(df1.head())
# # print(df2.head())

# import torch
# import csv
# import numpy as np

# def read_object_labels_csv(file, header=True):

#     multi_label = 0
#     remaining = 0
#     total = 0
#     images = []
#     num_categories = 0
#     print('[dataset] read', file)
#     with open(file, 'r') as f:
#         reader = csv.reader(f)
#         rownum = 0
#         for row in reader:
#             if header and rownum == 0:
#                 header = row
#             else:
#                 total += 1
#                 if num_categories == 0:
#                     num_categories = len(row) - 1
#                 name = row[0]
#                 labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
#                 labels = torch.from_numpy(labels)
#                 if labels.sum() >= 2:
#                     multi_label += 1
#                 else:
#                     remaining += 1
#                     item = (name, labels)
#                     images.append(item)
#             rownum += 1
#     print("Total : " + str(total))
#     print("Multi-label : " + str(multi_label))
#     print("Remaining : " + str(remaining))
#     return images, True if num_categories == 14 else False

# read_object_labels_csv('/home/yanjiexuan/multi-label-fsl/RC-Tran/idx/voc/classification_Train_novel.csv')
import pandas as pd


df = pd.read_csv('/home/yanjiexuan/multi-label-fsl/RC-Tran/idx/voc/classification_Train_novel.csv')
# print(df.head())

df1 = df[df['cat'] == 1]
# print(df1.head())

df1_ran = df1.sample(n=10,random_state=123,axis = 0)
print(df1_ran)