import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

feature_names = [feature_mapping.get(feature, feature) for feature in X_pass.columns]
explainer = shap.TreeExplainer(pass_model)
shap_values = explainer.shap_values(X_test_pass)
class_labels = [class_mapping[label] for label in pass_encoder.inverse_transform(range(len(class_mapping)))]
shap_values_mean = np.mean(np.abs(shap_values), axis=1) 
shap_values_df = pd.DataFrame(shap_values_mean, columns=feature_names, index=class_labels)
plt.figure(figsize=(12, 8))
sns.heatmap(
    shap_values_df.transpose(), 
    annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
    xticklabels=shap_values_df.index, 
    yticklabels=shap_values_df.columns
)
plt.title("Feature Contributions per Class", fontsize=16)
plt.xlabel("Class Labels", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.tight_layout()
plt.show()