import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random

# 设置随机种子以确保结果可复现
np.random.seed(42)
random.seed(42)

# ================ 1. 数据生成 ================
print("生成模拟数据集...")

# 定义数据集大小
n_samples = 10000

# 生成基础特征
age = np.random.normal(40, 10, n_samples).astype(int)
age = np.clip(age, 18, 75)  # 限制年龄在18-75岁之间

income = np.random.normal(60000, 20000, n_samples).astype(int)
income = np.clip(income, 20000, 200000)  # 限制收入范围

gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.5, 0.5])

satisfaction_score = np.random.randint(1, 6, n_samples)  # 1-5分满意度

# 创建DataFrame
df = pd.DataFrame({
    'age': age,
    'income': income,
    'gender': gender,
    'satisfaction_score': satisfaction_score
})

# 编码分类特征
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# 创建交互特征
df['age_income_ratio'] = df['age'] / (df['income'] / 1000)
df['income_satisfaction'] = df['income'] * df['satisfaction_score']

# 定义治疗效果（Treatment Effect）
# 假设：
# 1. 年龄较小且收入较高的客户对营销活动反应更好
# 2. 满意度低的客户更容易被营销活动打动
# 3. 女性比男性对营销活动更敏感

def calculate_true_uplift(row):
    # 基础响应概率
    base_response = 0.1  # 10%的基础转化率
    
    # 治疗效果 - 各种因素的组合
    treatment_effect = 0
    treatment_effect += 0.05 if row['age'] < 35 else 0  # 年轻人+5%
    treatment_effect += 0.08 if row['income'] > 70000 else 0  # 高收入+8%
    treatment_effect += 0.03 if row['gender'] == 1 else 0  # 女性+3%
    treatment_effect += 0.06 if row['satisfaction_score'] < 3 else 0  # 低满意度+6%
    
    # 交互效应
    if row['age'] < 35 and row['income'] > 70000:
        treatment_effect += 0.04  # 年轻高收入人群额外+4%
    
    return min(0.95, base_response + treatment_effect)  # 上限95%

# 分配实验组和对照组（随机分配，30%的概率进入实验组）
df['treatment'] = np.random.binomial(1, 0.3, n_samples)

# 计算真实的uplift
df['true_uplift'] = df.apply(calculate_true_uplift, axis=1)

# 生成结果变量y
# 对于对照组，使用基础转化率
# 对于实验组，使用基础转化率+治疗效果
def generate_response(row):
    if row['treatment'] == 1:
        # 实验组响应概率
        return 1 if random.random() < row['true_uplift'] else 0
    else:
        # 对照组响应概率
        base_response = 0.1  # 10%的基础转化率
        return 1 if random.random() < base_response else 0

df['converted'] = df.apply(generate_response, axis=1)

# ================ 2. 数据预处理 ================
print("数据预处理...")

# 定义特征列
feature_columns = ['age', 'income', 'gender', 'satisfaction_score', 'age_income_ratio', 'income_satisfaction']
X = df[feature_columns]
y = df['converted']
treatment = df['treatment']

# 划分训练集和测试集
X_train, X_test, y_train, y_test, treatment_train, treatment_test = train_test_split(
    X, y, treatment, test_size=0.3, random_state=42
)

# ================ 3. Uplift模型实现 ================
print("训练Uplift模型...")

# 方法1：Two Model Approach (TMA) - 分别训练实验组和对照组的模型
# 实验组模型
model_treatment = RandomForestClassifier(n_estimators=100, random_state=42)
model_treatment.fit(X_train[treatment_train==1], y_train[treatment_train==1])

# 对照组模型
model_control = RandomForestClassifier(n_estimators=100, random_state=42)
model_control.fit(X_train[treatment_train==0], y_train[treatment_train==0])

# 方法2：Class Variable Transformation (CVT) - 创建一个新的目标变量
# 对于实验组，目标变量为原始结果
# 对于对照组，目标变量为1-原始结果
y_cvt = np.where(treatment_train == 1, y_train, 1 - y_train)
model_cvt = GradientBoostingClassifier(n_estimators=100, random_state=42)
model_cvt.fit(X_train, y_cvt)

# ================ 4. 计算Uplift值 ================
print("计算Uplift值...")

# 使用TMA方法计算uplift
tma_uplift = model_treatment.predict_proba(X_test)[:, 1] - model_control.predict_proba(X_test)[:, 1]

# 使用CVT方法计算uplift（实际上是预测的概率）
cvt_uplift = model_cvt.predict_proba(X_test)[:, 1]

# 将结果添加到测试数据中
test_results = X_test.copy()
test_results['actual_conversion'] = y_test
test_results['treatment'] = treatment_test
test_results['tma_uplift_score'] = tma_uplift
test_results['cvt_uplift_score'] = cvt_uplift

# ================ 5. 模型评估 ================
print("评估模型性能...")

# 1. Uplift曲线和Qini系数
# 按uplift分数排序
test_results = test_results.sort_values('tma_uplift_score', ascending=False)
test_results['cumulative_customers'] = range(1, len(test_results) + 1)
test_results['cumulative_treatment_customers'] = test_results['treatment'].cumsum()
test_results['cumulative_control_customers'] = test_results['cumulative_customers'] - test_results['cumulative_treatment_customers']
test_results['cumulative_treatment_conversions'] = (test_results['treatment'] * test_results['actual_conversion']).cumsum()
test_results['cumulative_control_conversions'] = ((1 - test_results['treatment']) * test_results['actual_conversion']).cumsum()

test_results['uplift_at_k'] = (test_results['cumulative_treatment_conversions'] / test_results['cumulative_treatment_customers']
                               - test_results['cumulative_control_conversions'] / test_results['cumulative_control_customers'])

# 2. 计算Qini系数
def calculate_qini_coefficient(data):
    # 理想情况：完美的Uplift模型
    ideal = data.sort_values(['actual_conversion', 'treatment'], ascending=[False, False])
    ideal['cumulative_treatment_customers'] = ideal['treatment'].cumsum()
    ideal['cumulative_control_customers'] = ideal.index + 1 - ideal['cumulative_treatment_customers']
    ideal['cumulative_treatment_conversions'] = (ideal['treatment'] * ideal['actual_conversion']).cumsum()
    ideal['cumulative_control_conversions'] = ((1 - ideal['treatment']) * ideal['actual_conversion']).cumsum()
    ideal['qini_ideal'] = ideal['cumulative_treatment_conversions'] - (ideal['cumulative_treatment_customers'] / len(data)) * ideal['cumulative_control_conversions'].iloc[-1]
    
    # 实际情况
    data_sorted = data.sort_values('tma_uplift_score', ascending=False)
    data_sorted['cumulative_treatment_customers'] = data_sorted['treatment'].cumsum()
    data_sorted['cumulative_control_customers'] = data_sorted.index + 1 - data_sorted['cumulative_treatment_customers']
    data_sorted['cumulative_treatment_conversions'] = (data_sorted['treatment'] * data_sorted['actual_conversion']).cumsum()
    data_sorted['cumulative_control_conversions'] = ((1 - data_sorted['treatment']) * data_sorted['actual_conversion']).cumsum()
    data_sorted['qini_actual'] = data_sorted['cumulative_treatment_conversions'] - (data_sorted['cumulative_treatment_customers'] / len(data)) * data_sorted['cumulative_control_conversions'].iloc[-1]
    
    # 计算Qini系数（曲线下面积的差值）
    # 将index转换为numpy数组并计算sum
    qini_coef = (data_sorted['qini_actual'].sum() - (np.array(data_sorted.index) * (data_sorted['qini_actual'].iloc[-1] / len(data))).sum()) / len(data)
    return qini_coef

qini_coef = calculate_qini_coefficient(test_results)
print(f"Qini系数: {qini_coef:.4f}")

# 3. 按百分位数分组分析
test_results['percentile'] = pd.qcut(test_results['tma_uplift_score'], 10, labels=False)
percentile_analysis = test_results.groupby('percentile').agg({
    'actual_conversion': ['mean', 'count'],
    'treatment': 'mean',
    'tma_uplift_score': 'mean'
}).reset_index()

percentile_analysis.columns = ['percentile', 'conversion_rate', 'count', 'treatment_rate', 'avg_uplift_score']
print("\n按百分位数分组的转化率分析:")
print(percentile_analysis.sort_values('percentile', ascending=False))

# ================ 6. 可视化和结论分析 ================
print("\n生成可视化图表和结论分析...")

# 1. Uplift分布直方图
plt.figure(figsize=(12, 6))
sns.histplot(tma_uplift, bins=30, kde=True)
plt.title('Uplift分数分布')
plt.xlabel('Uplift分数')
plt.ylabel('频次')
plt.savefig('uplift_distribution.png')
plt.close()

# 2. Qini曲线
plt.figure(figsize=(12, 6))
plt.plot(test_results['cumulative_customers'] / len(test_results), 
         test_results['uplift_at_k'], label='Uplift曲线')
plt.plot([0, 1], [0, test_results['uplift_at_k'].iloc[-1]], 'r--', label='随机基线')
plt.title(f'Qini曲线 (Qini系数: {qini_coef:.4f})')
plt.xlabel('累积客户比例')
plt.ylabel('Uplift值')
plt.legend()
plt.grid(True)
plt.savefig('qini_curve.png')
plt.close()

# 3. 转化率与Uplift分数的关系
plt.figure(figsize=(12, 6))
sns.scatterplot(x='tma_uplift_score', y='actual_conversion', data=test_results, alpha=0.3)
plt.title('Uplift分数与实际转化率的关系')
plt.xlabel('Uplift分数')
plt.ylabel('实际转化率')
plt.grid(True)
plt.savefig('uplift_vs_conversion.png')
plt.close()

# ================ 7. 结论分析 ================
print("\n=== Uplift模型案例分析结论 ===")
print(f"1. 数据集概况：总样本数 {n_samples}，训练集 {len(X_train)}，测试集 {len(X_test)}")
print(f"2. 实验组占比：训练集 {treatment_train.mean():.2%}，测试集 {treatment_test.mean():.2%}")
print(f"3. 整体转化率：训练集 {y_train.mean():.2%}，测试集 {y_test.mean():.2%}")
print(f"4. Uplift模型性能：Qini系数 = {qini_coef:.4f}（值越高表示模型性能越好）")

# 分析高Uplift群体的特征
high_uplift_customers = test_results[test_results['tma_uplift_score'] > test_results['tma_uplift_score'].quantile(0.8)]
low_uplift_customers = test_results[test_results['tma_uplift_score'] < test_results['tma_uplift_score'].quantile(0.2)]

print("\n5. 高Uplift群体特征（前20%）：")
print(f"   - 平均年龄: {high_uplift_customers['age'].mean():.1f} 岁")
print(f"   - 平均收入: {high_uplift_customers['income'].mean():,.0f} 元")
print(f"   - 女性占比: {high_uplift_customers['gender'].mean():.2%}")
print(f"   - 平均满意度: {high_uplift_customers['satisfaction_score'].mean():.2f}/5 分")
print(f"   - 实际转化率: {high_uplift_customers['actual_conversion'].mean():.2%}")

print("\n6. 营销活动建议：")
print("   - 优先针对高Uplift群体（前20%）进行营销，预计可提升转化率")
print("   - 高Uplift客户特征：年轻（约35岁以下）、高收入（70000元以上）、满意度较低（3分以下）的女性客户")
print("   - 避免对低Uplift群体过度营销，以节省资源并提高ROI")

print("\n7. 模型改进方向：")
print("   - 可以尝试其他Uplift建模方法，如Uplift Random Forest或Meta-Learner")
print("   - 考虑添加更多交互特征或使用特征工程技术")
print("   - 使用A/B测试验证模型在实际场景中的效果")

print("\n可视化图表已保存：")
print("   - uplift_distribution.png：Uplift分数分布图")
print("   - qini_curve.png：Qini曲线图")
print("   - uplift_vs_conversion.png：Uplift分数与转化率关系图")