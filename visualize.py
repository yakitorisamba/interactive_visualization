import dash
from dash import dcc, html, Input, Output, callback, State
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

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(
        id='heatmap',
        figure=px.imshow(
            corr_matrix,
            text_auto=True,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns
        ),
        style={'width': '49%', 'display': 'inline-block'}
    ),
    html.Div(id='scatter-container', children=[], style={'width': '49%', 'display': 'inline-block'})
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
        figure=px.scatter(
            df, x=col_name, y=row_name, title=f"{col_name} vs {row_name}",
            labels={"x": col_name, "y": row_name},
            trendline="ols"  # 線形回帰線を追加
        )
    )
    
    # 子要素に新しい散布図を追加
    children.append(new_scatter)
    return children

if __name__ == '__main__':
    app.run_server(debug=True)
