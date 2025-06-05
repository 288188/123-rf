import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('RF.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "night_sleep_duration": {
        "type": "numerical", "min": 0.0, "max": 12.0, "default": 6.12  # in hours
    },
    "age": {
        "type": "numerical", "min": 45, "max": 100, "default": 63.48  # CHARLS focuses on middle-aged and elderly
    },
    "self_rated_health": {
        "type": "categorical", "options": [1, 2, 3, 4, 5], "default": 3  # 1=Very good, 5=Very poor
    },
    "life_satisfaction": {
        "type": "categorical", "options": [1, 2, 3, 4, 5], "default": 3  # 1=Very satisfied, 5=Very dissatisfied
    },
    "nap_duration": {
        "type": "numerical", "min": 0.0, "max": 180.0, "default": 42.34  # in minutes
    },
    "distance_vision": {
        "type": "categorical", "options": [1, 2, 3, 4, 5], "default": 2  # 1=Excellent, 5=Very poor
    },
    "education_level": {
        "type": "categorical", "options": [1, 2, 3, 4], "default": 1  # 1=No schooling, ..., 4=High school+
    },
    "household_size": {
        "type": "numerical", "min": 1, "max": 15, "default": 2.73
    },
    "near_vision": {
        "type": "categorical", "options": [1, 2, 3, 4, 5], "default": 2  # same scale as distance vision
    },
    "hearing_status": {
        "type": "categorical", "options": [1, 2, 3, 4, 5], "default": 2  # 1=Excellent, 5=Very poor
    },
    "hospitalizations_last_year": {
        "type": "numerical", "min": 0, "max": 10, "default": 0.27
    },
    "arthritis": {
        "type": "categorical", "options": [0, 1], "default": 0  # 0=No, 1=Yes
    },
    "residence_type": {
        "type": "categorical", "options": [0, 1], "default": 1  # 0=Rural, 1=Urban
    },
    "gender": {
        "type": "categorical", "options": [0, 1], "default": 0  # 0=Female, 1=Male
    },
    "hypertension": {
        "type": "categorical", "options": [0, 1], "default": 0  # 0=No, 1=Yes
    }
}



# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of AKI is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
