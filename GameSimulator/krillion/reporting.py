"""El Farol-style Plotly report for the minority answer game."""
import json
from pathlib import Path


def question_cells(result, p, q):
    r = result['rounds'][q]
    known = r['knowledge_before'][p]
    answers = sorted(set(known) | set(r['offered']))
    columns = [[] for _ in range(6)]
    for a in answers:
        index = r['offered'].index(a) if a in r['offered'] else None
        probability = r['answer_probabilities'][p][index] if index is not None else 0
        estimate = r['answer_estimates'][p][index] if index is not None else None
        values = [f'Route {a + 1}', 'Yes' if a in known else 'No',
                  'Yes' if index is not None else 'No',
                  f'{estimate:.3f}' if estimate is not None else '—',
                  f'{probability:.2%}',
                  f'{probability * (1-result["config"]["absence_probability"]):.2%}']
        for col, value in zip(columns, values):
            col.append(value)
    blank = result['config']['blank_probability']
    absent = result['config']['absence_probability']
    for values in (['Blank', '—', '—', '—', f'{blank:.2%}', f'{blank*(1-absent):.2%}'],
                   ['Absent', '—', '—', '—', '—', f'{absent:.2%}']):
        for col, value in zip(columns, values):
            col.append(value)
    return columns


def write_report(result, output, player=1):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    count = result['config']['players']
    if isinstance(player, bool) or not isinstance(player, int) or not 1 <= player <= count:
        raise ValueError('player must be an integer between 1 and the number of players')
    rounds = result['rounds']
    if not all('answer_probabilities' in r for r in rounds):
        raise ValueError('Rerun the simulation to record answer probabilities and knowledge snapshots')
    x = [r['round'] for r in rounds]
    p = player - 1
    fig = make_subplots(rows=5, cols=1, vertical_spacing=.065,
        row_heights=[.18, .15, .22, .31, .14],
        specs=[[{}], [{}], [{}], [{'type': 'table'}], [{}]],
        subplot_titles=['Cumulative score', 'Knowledge bank after each question',
            'Adjusted answer probabilities before each question · conditional on participation',
            'Question 1 · knowledge bank and probabilities before the decision',
            'Total attendance (all players)'])
    fig.add_trace(go.Scatter(x=[0]+x, y=[0]+[r['cumulative'][p] for r in rounds],
        name='Cumulative score', mode='lines', line=dict(color='#15803d'),
        hovertemplate='Round %{x}<br>Score %{y:.2f}<extra></extra>'), row=1, col=1)
    banks = []
    for r in rounds:
        bank = sorted(set(r['knowledge_before'][p]) | ({r['learned'][p]} if r['learned'][p] is not None else set()))
        banks.append(', '.join(f'Route {a+1}' for a in bank))
    fig.add_trace(go.Scatter(x=[0]+x,
        y=[len(result['initial_knowledge'][p])]+[r['knowledge_sizes'][p] for r in rounds],
        customdata=[', '.join(f'Route {a+1}' for a in result['initial_knowledge'][p])]+banks,
        name='Known answers', mode='lines', line=dict(color='#2563eb', shape='hv'),
        hovertemplate='Round %{x}<br>%{y} known answers<br>%{customdata}<extra></extra>'), row=2, col=1)
    slots = result['config']['answers_per_round']
    fig.add_trace(go.Heatmap(x=x, y=[f'Option {s+1}' for s in range(slots)],
        z=[[r['answer_probabilities'][p][s] for r in rounds] for s in range(slots)],
        customdata=[[[f'Route {r["offered"][s]+1}',
                      'Known' if r['offered'][s] in r['knowledge_before'][p] else 'Unknown']
                     for r in rounds] for s in range(slots)],
        zmin=0, zmax=1, colorscale='Blues', showscale=True,
        colorbar=dict(title='Probability', tickformat='.0%', len=.19, y=.545, thickness=12),
        hovertemplate='Question %{x}<br>%{customdata[0]} · %{customdata[1]}<br>P(answer | participates): %{z:.2%}<extra></extra>'), row=3, col=1)
    fig.add_trace(go.Table(columnwidth=[1.1, 1, .8, 1.1, 1.3, 1.3],
        header=dict(values=['Answer', 'Known before?', 'Offered?', 'Reward estimate',
                            'P(answer | participates)', 'P(answer overall)'],
                    fill_color='#e8eef5', align='left', height=32),
        cells=dict(values=question_cells(result, p, 0), fill_color='white',
                   align='left', height=25)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=[r['population'] for r in rounds],
        name='Total attendance', line=dict(color='#64748b')), row=5, col=1)
    for row in (1, 2, 5):
        fig.update_xaxes(title_text='Completed round', row=row, col=1)
    fig.update_xaxes(title_text='Question / round · hover for route identity; click to inspect', row=3, col=1)
    fig.update_yaxes(title_text='Total score', row=1, col=1)
    fig.update_yaxes(title_text='Known answers', rangemode='tozero', row=2, col=1)
    fig.update_yaxes(autorange='reversed', row=3, col=1)
    fig.update_yaxes(title_text='Attendees', range=[0, count+1], row=5, col=1)
    fig.update_layout(title=f'Minority game | Player {player}', template='plotly_white',
        height=1750, margin=dict(t=170, r=50, l=85, b=60),
        legend=dict(orientation='h', y=1.07, x=0),
        updatemenus=[dict(buttons=[dict(label=f'Player {i+1}', method='skip', name=f'player:{i}')
                                  for i in range(count)],
                         active=p, x=0, y=1.12, xanchor='left', yanchor='top'),
                     dict(buttons=[dict(label=f'Question {i+1}', method='skip', name=f'question:{i}')
                                   for i in range(len(rounds))],
                         active=0, x=.32, y=1.12, xanchor='left', yanchor='top')])
    script = Path(__file__).with_name('report.js').read_text(encoding='utf-8')
    script = script.replace('__RESULT__', json.dumps(result).replace('<', '\\u003c')).replace('__PLAYER__', str(p))
    page = fig.to_html(include_plotlyjs=True, full_html=True,
                       div_id='minority-chart', post_script=script)
    Path(output).write_text(page, encoding='utf-8')
    return fig
