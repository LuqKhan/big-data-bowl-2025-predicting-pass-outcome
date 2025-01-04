import random
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"

actual_pass_result = selected_play_data['passResult'].values[0]
acutal_outcome_full_label = class_mapping.get(actual_pass_result, "Unknown Outcome")
all_play_data = tracking_with_team[(tracking_with_team['gameId'] == game_id) & 
                                   (tracking_with_team['playId'] == play_id)]
sorted_frames = all_play_data['frameId'].unique()
sorted_frames.sort()
frames = []
for frame in sorted_frames:
    frame_data = []
    frame_tracking = all_play_data[all_play_data['frameId'] == frame]
    frame_data.append(go.Scatter(
        x=[line_of_scrimmage, line_of_scrimmage], 
        y=[0, 53.3], 
        mode='lines', 
        line=dict(color="blue", dash="dash"), 
        name="Scrimmage Line"
    ))
    frame_data.append(go.Scatter(
        x=[first_down_marker, first_down_marker], 
        y=[0, 53.3], 
        mode='lines', 
        line=dict(color="yellow", dash="dash"), 
        name="First Down Marker"
    ))
    for club in frame_tracking['club'].unique():
        club_data = frame_tracking[frame_tracking['club'] == club]
        color = colors.get(club, "grey") 
        frame_data.append(go.Scatter(
            x=club_data['x'], y=club_data['y'],
            mode='markers',
            marker=dict(color=color, size=10),
            name=club,
            text=club_data['displayName'],
            hoverinfo='text'
        ))
    frames.append(go.Frame(data=frame_data, name=str(frame)))
fig = go.Figure(
    data=frames[0].data, 
    layout=go.Layout(
        title={
            'text': f"Actual Play Animation: Week {random_week}, {home_team} vs {away_team}", 
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        annotations=[
            dict(
                x=0.5, y=1.1, xref="paper", yref="paper", 
                text=f"Actual Pass Result: {acutal_outcome_full_label}",
                showarrow=False,
                font=dict(size=16, color="black")
            )
        ],
        xaxis=dict(title="Field X Coordinate", range=[0, 120], showgrid=False),
        yaxis=dict(title="Field Y Coordinate", range=[0, 53.3], showgrid=False),
        plot_bgcolor="#00B140", 
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
            ]
        }]
    ),
    frames=frames
)

fig.frames = frames

fig.show()