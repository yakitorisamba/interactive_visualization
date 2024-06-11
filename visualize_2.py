import dash
from dash import dcc, html, Input, Output, State, callback
from jupyter_dash import JupyterDash
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

# データの準備
data = load_boston()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['TARGET'] = data.target

# 相関行列を計算
corr_matrix = df.corr()

app = JupyterDash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(
            id='heatmap',
            figure=px.imshow(
                corr_matrix,
                text_auto=True,
                labels=dict(color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns
            )
        )
    ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top'}),
    html.Div([
        html.Div(id='scatter-container', children=[]),
        html.Button('Plot Selected Data', id='plot-button', n_clicks=0)
    ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top'}),
    html.Div(id='plot-container', children=[])
])

@app.callback(
    Output('scatter-container', 'children'),
    Input('heatmap', 'clickData'),
    State('scatter-container', 'children')
)
def update_scatter(clickData, children):
    if clickData is None:
        return children

    # クリックされた列と行を取得
    points = clickData['points'][0]
    col_name = points['x']
    row_name = points['y']

    # 新しい散布図を生成
    new_scatter = dcc.Graph(
        id=f"graph-{len(children)}",
        figure=px.scatter(
            df, x=col_name, y=row_name, title=f"{col_name} vs {row_name}",
            labels={"x": col_name, "y": row_name},
            trendline="ols"  # 線形回帰線を追加
        )
    )

    # 子要素に新しい散布図を追加
    children.append(new_scatter)
    return children

def reshape_data(df, condition_columns):
    data = []
    for col in df.columns:
        if col.startswith("FRESH"):
            axis_num, level = col.split('_')[3].split('.')
            axis = axis_map[axis_num]
            for idx, value in df[col].items():
                row = {'Axis': float(axis), 'Level': str(level), 'value': value}
                for cond_col in condition_columns:
                    row[cond_col] = df.at[idx, cond_col]
                data.append(row)
    return pd.DataFrame(data)

@app.callback(
    Output('plot-container', 'children'),
    Input('plot-button', 'n_clicks'),
    State('scatter-container', 'children')
)
def plot_selected_data(n_clicks, children):
    if n_clicks == 0:
        return []

    selected_data = []
    for child in children:
        graph = child['props']['figure']
        selected_points = graph['data'][0]['selectedpoints']
        if selected_points:
            x_col = graph['layout']['xaxis']['title']['text']
            y_col = graph['layout']['yaxis']['title']['text']
            selected_data.append(df.iloc[selected_points][[x_col, y_col]])

    if not selected_data:
        return []

    merged_data = pd.concat(selected_data, axis=1)
    condition_columns = merged_data.columns.tolist()
    reshaped_data = reshape_data(merged_data, condition_columns)

    fig = px.line(reshaped_data, x='Axis', y='value', color='Level', facet_col='Level', facet_col_wrap=2)
    return dcc.Graph(figure=fig)

app.run_server(mode='inline', debug=True)
