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

feature_mapping = {
    'down': 'Down (1st, 2nd, etc.)',
    'yardsToGo': 'Yards to Go',
    'absoluteYardlineNumber': 'Field Position (Yard Line)',
    'expectedPoints': 'Expected Points',
    'preSnapHomeTeamWinProbability': 'Home Team Win Probability (Pre-Snap)',
    'preSnapVisitorTeamWinProbability': 'Visitor Team Win Probability (Pre-Snap)',
    'def_x_mean': 'Average Defensive X-Position',
    'def_x_std': 'Defensive X-Position Variability',
    'def_y_mean': 'Average Defensive Y-Position',
    'def_y_std': 'Defensive Y-Position Variability',
    'def_speed_mean': 'Average Defensive Speed',
    'def_accel_mean': 'Average Defensive Acceleration',
    'offenseFormation_encoded': 'Offensive Formation',
    'receiverAlignment_encoded': 'Receiver Alignment',
    'speed_accel_interaction': 'Speed-Acceleration Interaction (Defense)',
    'win_prob_yards_interaction': 'Win Probability-Yardage Interaction',
    'defensive_centroid_x': 'Defensive Centroid (X-Coordinate)',
    'defensive_centroid_y': 'Defensive Centroid (Y-Coordinate)',
    'proportion_motion_since_lineset': 'Proportion in Motion (Since Lineset)',
    'proportion_in_motion_at_snap': 'Proportion in Motion (At Snap)',
    'preSnapVisitorScore': 'Visitor Team Score (Pre-Snap)',
    'preSnapHomeScore': 'Home Team Score (Pre-Snap)',
    'pff_passCoverage_encoded': 'Pass Coverage Type',
    'avg_unblocked_pressure': 'Average Unblocked Defensive Pressure',
    'avg_time_to_throw': 'Average Time to Throw (QB)',
    'total_pass_defensed': 'Total Passes Defensed'
}
feature_importances = pass_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_pass.columns,
    'Importance': feature_importances
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df['Feature'] = importance_df['Feature'].map(feature_mapping)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance for Pass Outcome Prediction', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.tight_layout()
plt.show()