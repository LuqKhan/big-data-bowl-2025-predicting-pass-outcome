import random
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"

random_week = random.randint(6, 9)
tracking_file_path = f''
tracking_week_data = pd.read_csv(tracking_file_path)
tracking_with_team = tracking_week_data.merge(
    plays[['gameId', 'playId', 'possessionTeam', 'absoluteYardlineNumber', 'yardsToGo', 'playDescription']],
    on=['gameId', 'playId'],
    how='left'
)
pre_snap_data = tracking_with_team[
    (tracking_with_team['frameType'] == 'BEFORE_SNAP') 
]

random_play = pre_snap_data[['gameId', 'playId']].drop_duplicates().sample(1)
game_id, play_id = random_play['gameId'].values[0], random_play['playId'].values[0]
selected_tracking_data = pre_snap_data[(pre_snap_data['gameId'] == game_id) & 
                                       (pre_snap_data['playId'] == play_id)]
specific_frame = selected_tracking_data['frameId'].max()
single_frame_data = selected_tracking_data[selected_tracking_data['frameId'] == specific_frame]
selected_play_data = plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]
line_of_scrimmage = selected_play_data['absoluteYardlineNumber'].values[0]
first_down_marker = line_of_scrimmage + selected_play_data['yardsToGo'].values[0]
game_data = games[games['gameId'] == game_id]
home_team = game_data['homeTeamAbbr'].values[0]
away_team = game_data['visitorTeamAbbr'].values[0]
play_features = selected_play_data[pass_features]
predicted_outcome = pass_model.predict(play_features)[0]
predicted_outcome_label = pass_encoder.inverse_transform([predicted_outcome])[0]
colors = {
    'ARI': "#97233F", 'ATL': "#A71930", 'BAL': '#241773', 'BUF': "#00338D",
    'CAR': "#0085CA", 'CHI': "#C83803", 'CIN': "#FB4F14", 'CLE': "#311D00",
    'DAL': '#003594', 'DEN': "#FB4F14", 'DET': "#0076B6", 'GB': "#203731",
    'HOU': "#03202F", 'IND': "#002C5F", 'JAX': "#9F792C", 'KC': "#E31837",
    'LA': "#003594", 'LAC': "#0080C6", 'LV': "#000000", 'MIA': "#008E97",
    'MIN': "#4F2683", 'NE': "#002244", 'NO': "#D3BC8D", 'NYG': "#0B2265",
    'NYJ': "#125740", 'PHI': "#004C54", 'PIT': "#FFB612", 'SEA': "#69BE28",
    'SF': "#AA0000", 'TB': '#D50A0A', 'TEN': "#4B92DB", 'WAS': "#5A1414",
    'football': '#CBB67C'
}
fig = go.Figure(layout_yaxis_range=[0, 53.3], layout_xaxis_range=[0, 120])
for club in single_frame_data['club'].unique():
    club_data = single_frame_data[single_frame_data['club'] == club]
    color = colors.get(club, "grey")  
    fig.add_trace(go.Scatter(
        x=club_data['x'], y=club_data['y'],
        mode='markers',
        marker=dict(color=color, size=10),
        name=club,
        text=club_data['displayName'],
        hoverinfo='text'
    ))
fig.add_trace(go.Scatter(
    x=[line_of_scrimmage, line_of_scrimmage], y=[0, 53.3],
    mode='lines',
    name='Scrimmage Line',
    line=dict(color="blue", dash="dash")
))
fig.add_trace(go.Scatter(
    x=[first_down_marker, first_down_marker], y=[0, 53.3],
    mode='lines',
    name='First Down Marker',
    line=dict(color="yellow", dash="dash")
))
predicted_outcome_full_label = class_mapping.get(predicted_outcome_label, "Unknown Outcome")
fig.update_layout(
    title=f"Pre-Snap Setup: Week {random_week}, {home_team} vs {away_team} | Predicted Outcome: {predicted_outcome_full_label}",
    xaxis=dict(title="Field X Coordinate", showgrid=False),
    yaxis=dict(title="Field Y Coordinate", showgrid=False),
    plot_bgcolor="#00B140", 
    showlegend=True
)
fig.show()