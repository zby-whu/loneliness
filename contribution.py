import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

def statistic(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, y_test))

    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values[:, :, 0], X_test, plot_type='bar')
    shap.summary_plot(shap_values[:, :, 0], X_test)

    # 初始化SHAP解释器
    # explainer = shap.TreeExplainer(model)

    # 计算SHAP值
    # shap_values = explainer.shap_values(X_test)

    feature_importance = np.abs(shap_values[:, :, 0]).mean(0)
    sorted_indices = np.argsort(feature_importance)[::-1]
    top_ten_features = X_test.columns[sorted_indices[:30]]
    top_ten_importance = feature_importance[sorted_indices[:30]]

    # 计算贡献率
    total_importance = np.sum(feature_importance)
    contribution_ratio = top_ten_importance / total_importance

    # # 创建 DataFrame 保存贡献率占比数据
    # contribution_df = pd.DataFrame({'Feature': top_ten_features, 'Contribution Ratio': contribution_ratio})
    # # 保存为 CSV 文件
    # contribution_df.to_csv('contribution_ratio.csv', index=False)

    # 绘制贡献率柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(top_ten_features, contribution_ratio, color='skyblue')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Contribution Ratio', fontsize=12)
    plt.title('Thirty Features Contribution Ratio', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    for bar, ratio in zip(bars, contribution_ratio):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{ratio:.2%}', ha='center', va='bottom',
                 fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)  # 网格线置于最底层
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # # data = pd.read_csv('csv//4.26-2016-城.csv')
    # # data.dropna(inplace=True)
    # # data = data.replace([np.inf, -np.inf], np.nan)
    # # data.dropna(inplace=True)
    # # X = data.drop(columns=['loneliness'])
    # # y = data['loneliness']
    # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # # model = RandomForestClassifier(n_estimators=149, random_state=42)
    # # model.fit(X_train, y_train)
    # # statistic(model, X_test, y_test)
    #
    # # 加载数据集
    # data = pd.read_excel('D:/02-doctor24/12-ZBY/01-OLDER/1224 新版数据包含权重/2016-rural.xlsx')
    # data2 = pd.read_excel('D:/02-doctor24/12-ZBY/01-OLDER/1224 新版数据包含权重/2018-rural.xlsx')
    # data3 = pd.read_excel('D:/02-doctor24/12-ZBY/01-OLDER/1224 新版数据包含权重/2020-rural.xlsx')
    # data4 = pd.read_excel('D:/02-doctor24/12-ZBY/01-OLDER/1224 新版数据包含权重/2016-2020-rural.xlsx')
    # # data = pd.read_excel('D:/02-doctor24/12-ZBY/01-OLDER/1224 新版数据包含权重/2016-urban.xlsx')
    # # data2 = pd.read_excel('D:/02-doctor24/12-ZBY/01-OLDER/1224 新版数据包含权重/2018-urban.xlsx')
    # # data3 = pd.read_excel('D:/02-doctor24/12-ZBY/01-OLDER/1224 新版数据包含权重/2020-urban.xlsx')
    # # data4 = pd.read_excel('D:/02-doctor24/12-ZBY/01-OLDER/1224 新版数据包含权重/2016-2020-urban.xlsx')
    #
    # # 处理缺失值
    # data.dropna(inplace=True)
    # data2.dropna(inplace=True)
    # data3.dropna(inplace=True)
    # data4.dropna(inplace=True)
    #
    # # 处理无穷大值
    # data = data.replace([np.inf, -np.inf], np.nan)
    # data.dropna(inplace=True)
    # data2 = data2.replace([np.inf, -np.inf], np.nan)
    # data2.dropna(inplace=True)
    # data3 = data3.replace([np.inf, -np.inf], np.nan)
    # data3.dropna(inplace=True)
    # data4 = data4.replace([np.inf, -np.inf], np.nan)
    # data4.dropna(inplace=True)
    #
    # # 将因变量和自变量分开
    # X = data.drop(columns=['loneliness2'])
    # y = data['loneliness2']
    # X2 = data2.drop(columns=['loneliness2'])
    # y2 = data2['loneliness2']
    # X3 = data3.drop(columns=['loneliness2'])
    # y3 = data3['loneliness2']
    # X4 = data4.drop(columns=['loneliness2'])
    # y4 = data4['loneliness2']
    #
    # # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
    # X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)
    # X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=42)
    #
    # # 训练Gradient Boosting模型
    # model = RandomForestClassifier(n_estimators=149, random_state=42)
    # model.fit(X_train, y_train)
    # model2 = RandomForestClassifier(n_estimators=149, random_state=42)
    # model2.fit(X2_train, y2_train)
    # model3 = RandomForestClassifier(n_estimators=149, random_state=42)
    # model3.fit(X3_train, y3_train)
    # model4 = RandomForestClassifier(n_estimators=149, random_state=42)
    # model4.fit(X4_train, y4_train)
    #
    # # 从这开始 #####################################
    #
    # rf_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    # print(rf_auc)
    # rf_auc2 = roc_auc_score(y2_test, model2.predict_proba(X2_test)[:, 1])
    # print(rf_auc2)
    # rf_auc3 = roc_auc_score(y3_test, model3.predict_proba(X3_test)[:, 1])
    # print(rf_auc3)
    # rf_auc4 = roc_auc_score(y4_test, model4.predict_proba(X4_test)[:, 1])
    # print(rf_auc4)
    #
    # xxx = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # yyy = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #
    # fpr_rf, tpr_rf, threshold_rf = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    # fpr_rf2, tpr_rf2, threshold_rf2 = roc_curve(y2_test, model2.predict_proba(X2_test)[:, 1])
    # fpr_rf3, tpr_rf3, threshold_rf3 = roc_curve(y3_test, model3.predict_proba(X3_test)[:, 1])
    # fpr_rf4, tpr_rf4, threshold_rf4 = roc_curve(y4_test, model4.predict_proba(X4_test)[:, 1])
    # plt.plot(fpr_rf, tpr_rf, label="ROC-2016 : AUC = %.2f" % rf_auc)
    # plt.plot(fpr_rf2, tpr_rf2, label="ROC-2018 : AUC = %.2f" % rf_auc2)
    # plt.plot(fpr_rf3, tpr_rf3, label="ROC-2020 : AUC = %.2f" % rf_auc3)
    # plt.plot(fpr_rf4, tpr_rf4, label="ROC-2016~2020 : AUC = %.2f" % rf_auc4)
    # plt.plot(xxx, yyy, linestyle='--', color='black')
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.xlabel("FPR")
    # plt.ylabel("TPR(recall)")
    # plt.legend()
    # plt.show()
    #
    # statistic(model, X_test, y_test)
    # statistic(model2, X2_test, y2_test)
    # statistic(model3, X3_test, y3_test)
    # statistic(model4, X4_test, y4_test)

    data4 = pd.read_excel('D:/02-doctor24/12-ZBY/01-OLDER/1224 新版数据包含权重/2016-2020-urban-1231.xlsx')
    data4.dropna(inplace=True)
    data4 = data4.replace([np.inf, -np.inf], np.nan)
    data4.dropna(inplace=True)

    # 将因变量和自变量分开
    X4 = data4.drop(columns=['loneliness'])
    y4 = data4['loneliness']

    # 划分训练集和测试集
    X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=42)

    # 训练Gradient Boosting模型
    model4 = RandomForestClassifier(n_estimators=149, random_state=42)
    model4.fit(X4_train, y4_train)

    statistic(model4, X4_test, y4_test)