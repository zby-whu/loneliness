import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pycaret
from pycaret.classification import *

# 导入数据
file_path = '2016-city.csv'
# 读取数据，读取的所有数据名称为train
train = pd.read_csv(file_path)

# 输出所有数据的大小（矩阵大小）
print(train.shape)

# 数据类型
data_types = train.dtypes
print("数据类型：", data_types)
# 字符串类型的数量
string_types = data_types.apply(lambda dtype: dtype == object).sum()
print("字符串类型数量：", string_types)

# dropna是pandas的库，用于删除subset变量中的缺失值所在的行和列
train = train.dropna(subset=['loneliness'])
# 导入所需的模块并使用setup函数进行初始化，必须接受的参数为：pandas读取的数据帧和目标列的名称
s = setup(data=train, target='loneliness', remove_outliers=True, fix_imbalance=True, normalize=True, session_id=132)

# 进行模型的对比，indclude里面是需要对比的分类算法模型
# 直接使用compare_models()返回的是所有模型的最佳结果
# 使用compare_models(n_select = x)返回精度排名前x个的结果
# 使用include=[]，是可选的模型
# lr=logistic regression
# rf=random forest classifier
# lightgbm=light gradient boosting machine
best = compare_models()

# 输出最优模型
print(best)

# et = create_model('rf')
#
# plot_model(et, plot='auc')
#
# plot_model(et, plot='confusion_matrix')
#
# plot_model(et, plot='residuals')
#
# plot_model(et, plot='feature')

# 创建随机森林模型
et = create_model('rf')

# 模型比较，或者说展示模型性能
tuned_et, tuner = tune_model(et, return_tuner=True, n_iter=20, optimize='AUC')
print(tuner)

print(et)
print(tuned_et)

#plot_model(et, plot='auc')

interpret_model(et, plot = 'summary', save=True)

evaluate_model(et)

interpret_model(et, plot = 'summary', plot_type="bar")

interpret_model(et, plot = 'correlation', feature = 'age', interaction_index='gender')

evaluate_model(et)