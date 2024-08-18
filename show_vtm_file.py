from dash import Dash, dcc, html, Input, Output
import dash_vtk
import vtk

# VTKファイルのパス
input_filename = "input_multiblock.vtm"  # ここにMultiblockファイルのパスを指定

# Multiblockデータを読み込む
reader = vtk.vtkXMLMultiBlockDataReader()
reader.SetFileName(input_filename)
reader.Update()

multiblock_data = reader.GetOutput()

# 利用可能な物理量のリストを取得
available_arrays = []
for i in range(multiblock_data.GetNumberOfBlocks()):
    block = multiblock_data.GetBlock(i)
    if block is not None and block.GetPointData() is not None:
        for j in range(block.GetPointData().GetNumberOfArrays()):
            array_name = block.GetPointData().GetArrayName(j)
            if array_name not in available_arrays:
                available_arrays.append(array_name)

# Dashアプリケーションの作成
app = Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id="quantity-dropdown",
        options=[{'label': name, 'value': name} for name in available_arrays],
        value=available_arrays[0]  # デフォルトで最初の物理量を選択
    ),
    dash_vtk.View(id="vtk-view")
])

# コールバックでプルダウンの選択に応じて描画を更新
@app.callback(
    Output("vtk-view", "children"),
    Input("quantity-dropdown", "value")
)
def update_vtk_view(selected_quantity):
    append_filter = vtk.vtkAppendPolyData()

    for i in range(multiblock_data.GetNumberOfBlocks()):
        block = multiblock_data.GetBlock(i)
        if block is not None:
            if isinstance(block, vtk.vtkPolyData):
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(block)
                mapper.SetScalarModeToUsePointFieldData()
                mapper.SelectColorArray(selected_quantity)
                mapper.Update()
                append_filter.AddInputData(block)
            elif isinstance(block, vtk.vtkUnstructuredGrid):
                geometry_filter = vtk.vtkGeometryFilter()
                geometry_filter.SetInputData(block)
                geometry_filter.Update()

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(geometry_filter.GetOutput())
                mapper.SetScalarModeToUsePointFieldData()
                mapper.SelectColorArray(selected_quantity)
                mapper.Update()

                append_filter.AddInputData(geometry_filter.GetOutput())

    append_filter.Update()
    poly_data_output = append_filter.GetOutput()

    return dash_vtk.GeometryRepresentation([
        dash_vtk.PolyData(
            points=poly_data_output.GetPoints().GetData(),
            cells=poly_data_output.GetPolys().GetData(),
            pointData=poly_data_output.GetPointData().GetScalars().GetData(),
        ),
    ])

# アプリケーションの実行
if __name__ == "__main__":
    app.run_server(debug=True)
