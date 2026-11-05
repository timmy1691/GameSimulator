"""Interactive reports showing one selected player at a time."""
import numpy as np


def plot_player_report(result, output, title, player=1):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rounds, players = result.attendance.shape
    if isinstance(player, bool) or not isinstance(player, (int, np.integer)) or not 1 <= player <= players:
        raise ValueError('player must be an integer between 1 and the number of players')
    has_forecast = hasattr(result, 'forecasts')
    limits = np.broadcast_to(result.impulse_max, (players,)) if has_forecast else None
    impulse = has_forecast and np.any(limits > 0)
    def player_label(p):
        label = f'Player {p + 1}'
        if has_forecast:
            label += f' (max impulse: {limits[p]:g})'
        return label
    titles = ([] if not has_forecast else ['Forecast including own seat']) + [
        'Attendance probability before the decision', 'Decision: attend or stay home',
        'Cumulative reward', 'Total attendance (all players)']
    fig = make_subplots(rows=len(titles), cols=1, vertical_spacing=0.065, subplot_titles=titles)
    probability_row = 2 if has_forecast else 1
    decision_row, reward_row, total_row = probability_row + 1, probability_row + 2, probability_row + 3
    decision_rounds = list(range(1, rounds + 2))
    played_rounds = list(range(1, rounds + 1))
    cumulative = np.vstack([np.zeros((1, players)), np.cumsum(result.rewards, axis=0)])
    owners = []

    def add(values, x, name, row, owner, color, **kwargs):
        fig.add_trace(go.Scatter(x=x, y=np.asarray(values).tolist(), name=name,
                                visible=owner == player - 1, mode='lines',
                                line=dict(color=color, **kwargs),
                                hovertemplate='Round %{x}<br>%{y}<extra>%{fullData.name}</extra>'), row=row, col=1)
        owners.append(owner)

    for p in range(players):
        if has_forecast:
            add(result.forecasts[:, p] + 1, decision_rounds, 'Learned forecast', 1, p, '#2563eb')
            if impulse:
                add(result.decision_forecasts[:, p] + 1, played_rounds,
                    'Forecast after impulse', 1, p, '#ea580c', dash='dot')
        probabilities = result.probabilities
        if probabilities is None:
            probabilities = (result.forecasts + 1 <= result.capacity).astype(float)
        add(probabilities[:, p], decision_rounds, 'P(attend)', probability_row, p, '#7c3aed')
        add(result.attendance[:, p].astype(int), played_rounds, 'Decision', decision_row, p, '#0891b2', shape='hv')
        add(cumulative[:, p], list(range(rounds + 1)), 'Cumulative reward', reward_row, p, '#15803d')
    fig.add_trace(go.Scatter(x=played_rounds, y=result.attendance.sum(axis=1).tolist(),
                            name='Total attendance', line=dict(color='#64748b')), row=total_row, col=1)
    owners.append(None)
    if has_forecast:
        fig.add_hline(y=result.capacity, row=1, col=1, line_dash='dash', annotation_text='Capacity')
        fig.update_yaxes(title_text='Predicted attendees', range=[0, players + 1], row=1, col=1)
        fig.update_xaxes(title_text='Decision round (last forecast is for the next unplayed round)', row=1, col=1)
    fig.add_hline(y=result.capacity, row=total_row, col=1, line_dash='dash', annotation_text='Capacity')
    fig.add_hline(y=0, row=reward_row, col=1, line_color='#cbd5e1')
    fig.update_yaxes(title_text='P(attend)', range=[-0.05, 1.05], row=probability_row, col=1)
    fig.update_xaxes(title_text='Decision round (last probability is for the next unplayed round)', row=probability_row, col=1)
    fig.update_yaxes(tickvals=[0, 1], ticktext=['Stay home', 'Attend'], range=[-0.1, 1.1], row=decision_row, col=1)
    fig.update_yaxes(title_text='Total reward', row=reward_row, col=1)
    fig.update_yaxes(title_text='Attendees', range=[0, players + 1], row=total_row, col=1)
    for row in (decision_row, reward_row, total_row):
        fig.update_xaxes(title_text='Completed round' if row == reward_row else 'Round', row=row, col=1)
    buttons = [dict(label=player_label(p), method='update', args=[
        {'visible': [owner is None or owner == p for owner in owners]},
        {'title': {'text': f'{title} | {player_label(p)}'}}]) for p in range(players)]
    fig.update_layout(title=f'{title} | {player_label(player - 1)}', template='plotly_white',
                      height=1450 if has_forecast else 1200, margin=dict(t=170, r=50, l=85),
                      legend=dict(orientation='h', y=1.07, x=0),
                      updatemenus=[dict(buttons=buttons, active=player - 1, x=0, y=1.12,
                                        xanchor='left', yanchor='top')])
    fig.write_html(str(output), include_plotlyjs=True, auto_open=False)
    return fig
