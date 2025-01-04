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

games = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/games.csv')
plays = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv')
player_play = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/player_play.csv')
tracking_files = ['/kaggle/input/nfl-big-data-bowl-2025/tracking_week_1.csv',
                  '/kaggle/input/nfl-big-data-bowl-2025/tracking_week_2.csv',
                  '/kaggle/input/nfl-big-data-bowl-2025/tracking_week_3.csv',
                  '/kaggle/input/nfl-big-data-bowl-2025/tracking_week_4.csv',
                  '/kaggle/input/nfl-big-data-bowl-2025/tracking_week_5.csv']
tracking_data = pd.concat([pd.read_csv(file) for file in tracking_files], ignore_index=True)
tracking_with_team = tracking_data.merge(
    plays[['gameId', 'playId', 'possessionTeam', 'pff_passCoverage', 'passResult']],
    on=['gameId', 'playId'],
    how='left'
)
defensive_tracking = tracking_with_team[tracking_with_team['club'] != tracking_with_team['possessionTeam']]
defense_summary = defensive_tracking.groupby(['gameId', 'playId']).agg({
    'x': ['mean', 'std'],
    'y': ['mean', 'std'], 
    's': 'mean',          
    'a': 'mean'            
}).reset_index()
defense_summary.columns = ['gameId', 'playId', 'def_x_mean', 'def_x_std', 'def_y_mean', 'def_y_std', 'def_speed_mean', 'def_accel_mean']
plays = plays.merge(defense_summary, on=['gameId', 'playId'], how='left')
defensive_centroid = defensive_tracking.groupby(['gameId', 'playId']).agg({'x': 'mean', 'y': 'mean'}).reset_index()
defensive_centroid.columns = ['gameId', 'playId', 'defensive_centroid_x', 'defensive_centroid_y']
plays = plays.merge(defensive_centroid, on=['gameId', 'playId'], how='left')
plays['speed_accel_interaction'] = plays['def_speed_mean'] * plays['def_accel_mean']
plays['win_prob_yards_interaction'] = plays['preSnapHomeTeamWinProbability'] * plays['yardsToGo']
motion_features = player_play.groupby(['gameId', 'playId']).agg({
    'motionSinceLineset': 'mean',
    'inMotionAtBallSnap': 'mean'
}).reset_index()
motion_features.rename(columns={
    'motionSinceLineset': 'proportion_motion_since_lineset',
    'inMotionAtBallSnap': 'proportion_in_motion_at_snap'
}, inplace=True)
plays = plays.merge(motion_features, on=['gameId', 'playId'], how='left')
plays['offenseFormation'] = plays['offenseFormation'].dropna()
offense_encoder = LabelEncoder()
plays['offenseFormation_encoded'] = offense_encoder.fit_transform(plays['offenseFormation'])

plays['receiverAlignment'] = plays['receiverAlignment'].dropna()
alignment_encoder = LabelEncoder()
plays['receiverAlignment_encoded'] = alignment_encoder.fit_transform(plays['receiverAlignment'])

plays['pff_passCoverage'] = plays['pff_passCoverage'].dropna()
coverage_encoder = LabelEncoder()
plays['pff_passCoverage_encoded'] = coverage_encoder.fit_transform(plays['pff_passCoverage'])
scaler = StandardScaler()
plays[['def_x_mean', 'def_x_std', 'def_y_mean', 'def_y_std', 'def_speed_mean', 
       'def_accel_mean', 'speed_accel_interaction', 'win_prob_yards_interaction',
       'defensive_centroid_x', 'defensive_centroid_y', 
       'proportion_motion_since_lineset', 'proportion_in_motion_at_snap']] = scaler.fit_transform(
    plays[['def_x_mean', 'def_x_std', 'def_y_mean', 'def_y_std', 'def_speed_mean', 
           'def_accel_mean', 'speed_accel_interaction', 'win_prob_yards_interaction',
           'defensive_centroid_x', 'defensive_centroid_y', 
           'proportion_motion_since_lineset', 'proportion_in_motion_at_snap']]
)
coverage_features = [
    'down', 'yardsToGo', 'absoluteYardlineNumber',
    'expectedPoints', 'preSnapHomeTeamWinProbability',
    'preSnapVisitorTeamWinProbability', 'def_x_mean',
    'def_x_std', 'def_y_mean', 'def_y_std', 'def_speed_mean',
    'def_accel_mean', 'offenseFormation_encoded',
    'receiverAlignment_encoded', 'speed_accel_interaction',
    'win_prob_yards_interaction', 'defensive_centroid_x', 'defensive_centroid_y',
    'proportion_motion_since_lineset', 'proportion_in_motion_at_snap', 
    'preSnapVisitorScore', 'preSnapHomeScore', 'pff_passCoverage_encoded'
]
pass_defensed_summary = player_play.groupby(['gameId', 'playId'])['passDefensed'].sum().reset_index()
pass_defensed_summary.rename(columns={'passDefensed': 'total_pass_defensed'}, inplace=True)
plays = plays.merge(pass_defensed_summary, on=['gameId', 'playId'], how='left')
plays['total_pass_defensed'] = plays['total_pass_defensed'].dropna()
plays['unblockedPressure'] = plays['unblockedPressure'].map({True: 1, False: 0})
plays['unblockedPressure'] = pd.to_numeric(plays['unblockedPressure'], errors='coerce')
unblocked_pressure_summary = plays.groupby(['gameId', 'playId'])['unblockedPressure'].mean().reset_index()
unblocked_pressure_summary.rename(columns={'unblockedPressure': 'avg_unblocked_pressure'}, inplace=True)
time_to_throw_summary = plays.groupby(['gameId', 'playId'])['timeToThrow'].mean().reset_index()
time_to_throw_summary.rename(columns={'timeToThrow': 'avg_time_to_throw'}, inplace=True)
plays = plays.merge(unblocked_pressure_summary, on=['gameId', 'playId'], how='left')
plays = plays.merge(time_to_throw_summary, on=['gameId', 'playId'], how='left')
plays['avg_time_to_throw'] = pd.to_numeric(plays['avg_time_to_throw'], errors='coerce')
plays['avg_unblocked_pressure'] = pd.to_numeric(plays['avg_unblocked_pressure'], errors='coerce')
plays['avg_time_to_throw'].dropna()
plays['avg_unblocked_pressure'].dropna()
pass_features = coverage_features + [
    'avg_unblocked_pressure',
    'avg_time_to_throw',
    'total_pass_defensed',
]
pass_target = 'passResult'
plays['passResult'] = plays['passResult'].fillna('Run')
missing_values = plays[pass_features].isnull().sum()
plays[pass_features] = plays[pass_features].fillna(plays[pass_features].mean())
pass_encoder = LabelEncoder()
pass_encoder.fit(['Run', 'C', 'I', 'IN', 'S', 'R']) 
plays['passResult_encoded'] = pass_encoder.transform(plays['passResult'])
filtered_plays = plays.dropna(subset=pass_features + ['passResult_encoded'])
y_pass = filtered_plays['passResult_encoded']
X_pass = filtered_plays[pass_features]
y_pass = filtered_plays['passResult_encoded']
smote = SMOTE(random_state=42)
X_pass_resampled, y_pass_resampled = smote.fit_resample(X_pass, y_pass)
X_train_pass, X_test_pass, y_train_pass, y_test_pass = train_test_split(
    X_pass_resampled, y_pass_resampled, test_size=0.2, random_state=42, stratify=y_pass_resampled
)
pass_model = LGBMClassifier(
    random_state=42,
    n_estimators=500,
    learning_rate=0.03,
    max_depth=15,
    class_weight='balanced',
    verbosity=-1
)
pass_model.fit(X_train_pass, y_train_pass)
y_pred_pass = pass_model.predict(X_test_pass)
pass_accuracy = accuracy_score(y_test_pass, y_pred_pass)
unique_classes = np.unique(y_test_pass)
report_dict = classification_report(
    y_test_pass,
    y_pred_pass,
    labels=unique_classes,
    target_names=pass_encoder.inverse_transform(unique_classes),
    output_dict=True
)
class_mapping = {
    "Run": "Run Play",
    "C": "Completion",
    "I": "Incomplete",
    "IN": "Interception",
    "S": "Quarterback Sack",
    "R": "Scramble"
}
report_df = pd.DataFrame(report_dict).transpose()
report_df.rename(index=class_mapping, inplace=True)
formatted_report = report_df[['precision', 'recall', 'f1-score', 'support']].round(2)

bar_data = formatted_report.iloc[:-1, :-1].reset_index() 
bar_data = pd.melt(bar_data, id_vars='index', var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 8))
sns.barplot(
    data=bar_data,
    x='index', y='Score', hue='Metric', palette='viridis'
)

plt.title("Classification Metrics per Class", fontsize=16)
plt.xlabel("Class Labels", fontsize=14)
plt.ylabel("Score (%)", fontsize=14)

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) 
plt.gca().set_yticks([0.8, 0.9, 1.0]) 

plt.xticks(rotation=45, ha='right')
plt.legend(title='Metrics', loc='lower right')
plt.tight_layout()
plt.show()