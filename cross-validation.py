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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    pass_model,
    X_pass_resampled,
    y_pass_resampled,
    cv=skf,
    scoring='accuracy'
)
pass_model.fit(X_train_pass, y_train_pass)
y_pred_pass = pass_model.predict(X_test_pass)
pass_accuracy = accuracy_score(y_test_pass, y_pred_pass)
report_dict = classification_report(
    y_test_pass,
    y_pred_pass,
    labels=np.unique(y_test_pass),
    target_names=pass_encoder.inverse_transform(np.unique(y_test_pass)),
    output_dict=True
)
report_df = pd.DataFrame(report_dict).transpose()
report_df.rename(index=class_mapping, inplace=True)
formatted_report = report_df[['precision', 'recall', 'f1-score', 'support']].round(2)
formatted_report.reset_index(inplace=True)
formatted_report = formatted_report.rename(columns={"index": "Pass Outcome"})
metrics_data = formatted_report.loc[:5, ["Pass Outcome", "precision", "recall", "f1-score"]]
metrics_data_long = metrics_data.melt(id_vars=["Pass Outcome"], 
                                      var_name="Metric", 
                                      value_name="Score")
plt.figure(figsize=(12, 8))
sns.barplot(
    data=metrics_data_long,
    x="Pass Outcome", 
    y="Score", 
    hue="Metric",
    palette="coolwarm"
)
plt.title("Precision, Recall, and F1-Score for Pass Outcomes", fontsize=16)
plt.xlabel("Pass Outcome", fontsize=14)
plt.ylabel("Score (%)", fontsize=14)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) 
plt.gca().set_yticks([0.8, 0.9, 1.0]) 
plt.legend(title="Metric", loc="upper right", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()