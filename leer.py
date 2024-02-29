#Importacion de librerias para la conexion con la API de Google Sheets
import os
import base64
import json

#Importacion de pandas y numpy
import pandas as pd
import numpy as np

#Librerias de dash
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import dash_table
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask import Flask, render_template


#Librerias para la regresión lineal
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

#Librerias para el formato de la hora
from datetime import datetime
from datetime import timedelta

# Librerias para el consumo de la Google Sheets
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.oauth2 import service_account


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
""" SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
KEY = 'key.json'
SPREADSHEETS_ID = '1AZIGMqBmzAqVZzGbZDQdKUK_R9xVk7iuPrZ-0es5Bi4'

creds = None
creds = service_account.Credentials.from_service_account_file(KEY, scopes=SCOPES)

service = build('sheets', 'v4', credentials=creds)
sheets = service.spreadsheets() """

# Constantes
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SPREADSHEETS_ID = '1AZIGMqBmzAqVZzGbZDQdKUK_R9xVk7iuPrZ-0es5Bi4'

# Decodificar las credenciales codificadas en base64 de la variable de entorno
creds_json = base64.b64decode(os.environ['GOOGLE_SHEETS_CREDS_BASE64']).decode('utf-8')
creds_dict = json.loads(creds_json)

# Carga las credenciales desde el diccionario
creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=SCOPES)

# Construye el servicio utilizando las credenciales
service = build('sheets', 'v4', credentials=creds)
sheets = service.spreadsheets()

#Obtencion de Datos de la Hoja 1-----------------------------------------------------------------------------------------------------------------------------------------

result = sheets.values().get(spreadsheetId=SPREADSHEETS_ID, range='Hoja 1!A:H').execute()
values = result.get('values', [])
df = pd.DataFrame(values[1:], columns=values[0])
df['Fecha y Hora'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora'])
df[['Temperatura', 'Humedad', 'CO2','Punto', 'Latitud', 'Longitud']] = df[['Temperatura', 'Humedad', 'CO2','Punto', 'Latitud', 'Longitud']].astype(float)
dias_unicos = df['Fecha'].unique()
data_count_df = len(df)

#Obtencion de Datos de la Hoja 2-----------------------------------------------------------------------------------------------------------------------------------------
result2 = sheets.values().get(spreadsheetId=SPREADSHEETS_ID, range='Hoja 2!A:E').execute()
values2 = result2.get('values', [])
start_month = None
end_month = None
if values2:
    # Las columnas de la hoja de cálculo deben coincidir con este orden
    expected_columns = ['Fecha Mes', 'Pendientes', 'Punto Medicion', 'Latitud', 'Longitud']
    
    # Verificar si la primera fila de values2 coincide con los nombres de columna esperados
    if values2[0] == expected_columns:
        df2 = pd.DataFrame(values2[1:], columns=values2[0])
        # También puedes convertir la columna 'Fecha Mes' a tipo datetime
        df2['Fecha Mes'] = pd.to_datetime(df2['Fecha Mes'])
    else:
        # Manejar el caso en que las columnas no coincidan
        df2 = pd.DataFrame(columns=expected_columns)
else:
    # Manejar el caso en que no se encuentren datos en la hoja 2
    df2 = pd.DataFrame(columns=['Fecha Mes', 'Pendientes', 'Punto Medicion', 'Latitud', 'Longitud'])
data_count_df2 = len(df2)
fechas_unicas = df2['Fecha Mes'].dt.strftime('%m/%d/%Y').drop_duplicates().values

#API para el mapa----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

px.set_mapbox_access_token('pk.eyJ1IjoiZGF2aWRhZHJpZWwiLCJhIjoiY2xtbXoyZWh5MHB4ejJxcXNsdW5sc3BneSJ9.hg7yqKdO_lRO0UfCrdRBDA')

#App Layout------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
app = dash.Dash(__name__)
app.title = 'Medidor CO2'
app.layout = html.Div(children=[
    html.Link(
        rel='stylesheet',
        href='https://fonts.googleapis.com/css2?family=Manrope&family=Open+Sans&display=swap"'),
    
    html.Div(id='header', children=[
        html.H1('Mediciones de CO2', id='title'),
    ]),
    
    html.Div(id='content', style={'display': 'flex'}, children=[
        # Div para todo el contenido
        html.Div(id='main-content', 
                 children=[
            html.P('A continuación se presenta el registro de lecturas del Medidor de CO2 para la investigación de CO2 del suelo volcánico.', style={'font-size': '14px','font-weight': '700', 'text-align': 'center'}),
            
            html.Div(id='indicacion',children=[
                html.Div(id='description', children=[
                    html.P('1. Seleccione la fecha para ver las lecturas correspondientes al dia'),
                    html.P('2. Seleccione el punto de medicion para ver las lecturas correspondientes a este'),
                ]), # Eliminar el estilo y añadir el id
                html.Div(id='filter-container',children=[
                    dcc.Dropdown(
                        id='filtro-dia',
                        options=[{'label': dia, 'value': dia} for dia in dias_unicos],
                        value=dias_unicos[0],
                        clearable=False,
                        className='custom-dropdown'
                    ),
                    dcc.Dropdown(
                        id='filtro-punto',
                        options=[],  # Inicialmente vacío
                        value=None,
                        className='custom-dropdown'
                    ),  
                    
                ]), # Eliminar el estilo y añadir el id
            ]),
            html.Div(id='counts', children=[
                        html.P(id='hoja1', className="data-count"),
                        html.P(id='hoja2', className="data-count"),
                    ]), 
            html.P('Los siguientes graficos muestran las lecturas de concentracion de CO2, temperatura y humedad en el punto seleccionado del filtro. En el mapa se pueden observar en marcadores azules los puntos de medicion correspondientes al dia seleccionado en el filtro.'
                  ),
            
            dcc.Graph(
                id='mapa-ubicacion-actual',
            ),
            
            html.Div(id='graph-row', children=[
                dcc.Graph(
                    id='co2-hora-graph',
                ),
                dcc.Graph(
                    id='temperatura-humedad-graph',
                ),
            ]),
                        
            html.Div(id='regression-controls', children=[
                html.P('Seleccione el intervalo de tiempo donde la grafica de concentracion de CO2 tenga una tendencia lineal, ingrese un limite inferior y superior en formato HH:MM:SS que pertenezcan a la linea (use el cursor en la linea para ver las coordenadas de los puntos) y aplique la Regresión Lineal. Al realizar este procedimiento el perfil de flujo de co2 vs distancia se actualizara para el punto y fecha seleccionado', style={'font-size': '14px', 'text-align': 'justify','margin':'20px'}),
                html.Div(id='regression-inputs', className='container', children=[
                    html.Div(id='regression-inputs-left', className='left-container', children=[
                        dcc.Input( 
                            id='input-inicio-hora', type='text',
                            placeholder='Hora de inicio (HH:MM:SS)',
                        ),
                    ]),
                    html.Div(id='regression-inputs-center', className='center-container', children=[
                        dcc.Input(
                            id='input-fin-hora', type='text',
                            placeholder='Hora de fin (HH:MM:SS)',
                        ),
                    ]),
                    html.Div(id='regression-inputs-right', className='right-container', children=[
                        html.Button(
                            'Aplicar Regresión Lineal',
                            id='btn-regresion',
                        ),
                    ]),
                ]),
            ]),
            
            
                html.Div(id='table-and-graph-container', style={'display': 'flex'}, children=[
                    html.Div(id='table-container', children=[
                        dash_table.DataTable(
                            id='tabla-co2',
                            columns=[{"name": i, "id": i} for i in ['Recuento', 'Hora', 'CO2']],
                            style_cell={'textAlign': 'center', 'font-family':'Manrope', 'font-size':'10px'},
                            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                            style_table={'maxHeight': '300px', 'overflowY': 'scroll'}
                        ),
                    ]),
                    html.Div(id='graph-container', children=[
                        dcc.Graph(
                            id='regresion-lineal-graph',
                        ),
                    ]),
                ]),
                
            

            html.Div(id='monthly-selector', children=[
                html.P('Seleccione los meses en los que desea ver el perfil de mediciones de flujo vs distancia.', style={'font-size': '14px', 'text-align': 'justify','margin': '20px'}),
                dcc.Store(id='df2-store', data=df2.to_dict('records')),
                dcc.Dropdown(
                    id='fecha-selector',
                    options=[{'label': fecha, 'value': fecha} for fecha in fechas_unicas],
                    multi=True,  # Permite selección múltiple
                    className='custom-dropdown'
                ),
                
                dcc.Graph(id='grafico-co2-vs-distancia'),
            ]),
            
            html.Div(id='update-button-container', children=[
                html.Button('Actualizar datos', id='btn-actualizar-2', style={'font-size': '14px'}),
            ]),
            
        ]),
    ]),
])

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def create_temperatura_humedad_graph(df_filtrado,nombre_punto):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtrado['Fecha y Hora'], y=df_filtrado['Temperatura'], name='Temperatura', yaxis='y1'))
    fig.add_trace(go.Scatter(x=df_filtrado['Fecha y Hora'], y=df_filtrado['Humedad'], name='Humedad', yaxis='y2'))
    fig.update_layout(
        title=f'Temperatura y Humedad - Punto: {nombre_punto}',
        font=dict(family='Manrope', size=10,color='black'), 
        plot_bgcolor='white',
        yaxis=dict(title='Temperatura (°C)', side='left', showgrid=True),
        yaxis2=dict(title='Humedad (%)', side='right', overlaying='y', showgrid=True),
        xaxis=dict(title='Fecha y Hora', showgrid=True),
        margin=dict(l=10, r=10, t=50, b=25),
        showlegend=False
        )
    fig.update_xaxes(gridcolor='lightgrey', linewidth=1)
    fig.update_yaxes(gridcolor='lightgrey', linewidth=1)
    return fig

def create_co2_hora_graph(intervalo_co2_hora, pendiente, intercepto):
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtrado['Fecha y Hora'], 
        y=df_filtrado['CO2'], 
        name=f'CO2 - Punto: {nombre_punto}', 
        line=dict(color='black', width=1.5))
        )
    fig.update_layout(
        title=f'CO2 vs Hora - Punto: {nombre_punto}', 
        font=dict(family='Manrope', size=10,color='black'),
        plot_bgcolor='white',
        xaxis=dict(title='Fecha y Hora', showgrid=True), 
        yaxis=dict(title='CO2', showgrid=True),
        margin=dict(l=10, r=10, t=50, b=25),
        )
    fig.update_xaxes(gridcolor='lightgrey', linewidth=1)
    fig.update_yaxes(gridcolor='lightgrey', linewidth=1)
    return fig

def create_regresion_lineal_graph(df, pendiente):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Recuento'], 
        y=df['CO2'], 
        mode='markers', 
        name='CO2',
        line=dict(color='black', width=1.5))
        )
    fig.add_trace(go.Scatter(
        x=df['Recuento'], 
        y=df['y_pred'], 
        mode='lines', 
        name='Regresión Lineal')
        )
    fig.update_layout(
        title='Regresión Lineal CO2 vs Recuento',
        font=dict(family='Manrope', size=10,color='black'),
        plot_bgcolor='white',
        xaxis=dict(title='Recuento',showgrid=True,),
        yaxis=dict(title='CO2',showgrid=True,), 
        annotations=[
            go.layout.Annotation(
                x=1,y=1,
                xref="paper",
                yref="paper",
                text=f' Ecuacion: y = {pendiente:.2f}t + {intercepto:.2f} ',
                showarrow=False,font=dict(size=11),
            )
        ],
        margin=dict(l=10, r=10, t=50, b=25),
    )
    fig.update_xaxes(gridcolor='lightgrey', linewidth=1)
    fig.update_yaxes(gridcolor='lightgrey', linewidth=1)
    return fig


@app.callback(
    Output('filtro-dia', 'options'),
    Output('filtro-dia', 'value'),
    Input('btn-actualizar-2', 'n_clicks')
)
def actualizar_datos(n_clicks):
    global df
    global data_count_df
    # Actualizar datos de la Hoja 1
    result = sheets.values().get(spreadsheetId=SPREADSHEETS_ID, range='Hoja 1').execute()
    values = result.get('values', [])
    df = pd.DataFrame(values[1:], columns=values[0])
    df['Fecha y Hora'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora'])
    df[['Temperatura', 'Humedad', 'CO2', 'Punto', 'Latitud', 'Longitud']] = df[['Temperatura', 'Humedad', 'CO2', 'Punto', 'Latitud', 'Longitud']].astype(float)
    dias_unicos = df['Fecha'].unique()

    opciones = [{'label': dia, 'value': dia} for dia in dias_unicos]
    valor = dias_unicos[0]
    data_count_df = len(df)
    return opciones, valor


@app.callback(
    [Output('hoja1', 'children'),
     Output('temperatura-humedad-graph', 'figure'),
     Output('mapa-ubicacion-actual', 'figure'),
     Output('co2-hora-graph', 'figure')],
    [Input('filtro-dia', 'value'),
     Input('filtro-punto', 'value')]
)
def update_graphs(dia_seleccionado, punto_seleccionado):
    if dia_seleccionado:     
        df_filtrado = df[df['Fecha'] == dia_seleccionado]
        texto_marcadores = [f'Punto: {punto}' for punto in df_filtrado['Punto']]
        mapa = go.Figure(
            data=go.Scattermapbox(
                lat = df_filtrado['Latitud'],
                lon = df_filtrado['Longitud'],
                mode = 'markers',
                marker = go.scattermapbox.Marker(
                    size = 10,
                    color = 'blue'
                ),
                text=texto_marcadores, 
                textposition="top right",
            ),
            
            layout = go.Layout(
                title_text = 'Coordenadas de mediciones',
                font=dict(family='Manrope', size=12),
                autosize=True,
                hovermode='closest',
                showlegend=False,
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoiZGF2aWRhZHJpZWwiLCJhIjoiY2xtbXoyZWh5MHB4ejJxcXNsdW5sc3BneSJ9.hg7yqKdO_lRO0UfCrdRBDA',
                    bearing=0,
                    center=dict(
                        lat=13.5939, 
                        lon=-89.3335
                    ),
                    pitch=0,
                    zoom=8,
                    style='outdoors'
                ),
                margin=dict(l=20, r=20, t=50, b=25),
            )
        )
        if punto_seleccionado:
            df_filtrado = df_filtrado[df_filtrado['Punto'] == punto_seleccionado]
            
        nombre_punto = punto_seleccionado if punto_seleccionado else 'Todos los Puntos'  # Nombre del punto seleccionado o "Todos los Puntos" si no se selecciona ninguno
        temperatura_humedad_graph = create_temperatura_humedad_graph(df_filtrado, nombre_punto)  # Pasamos el nombre del punto a la función
        co2_hora_graph = create_co2_hora_graph(df_filtrado, nombre_punto)  # Pasamos el nombre del punto a la función
       
    else:
        # Si no hay fecha seleccionada, muestra gráficos vacíos o algún mensaje de error
        temperatura_humedad_graph = go.Figure()
        mapa = go.Figure()
        co2_hora_graph = go.Figure()
        nombre_punto = 'Todos los Puntos' 
    
    return f"Lecturas de Sensores\n {data_count_df}",temperatura_humedad_graph, mapa,co2_hora_graph



@app.callback(
    Output('filtro-punto', 'options'),
    Input('filtro-dia', 'value')
)
def update_punto_options(dia_seleccionado):
    if dia_seleccionado:
        puntos_disponibles = df[df['Fecha'] == dia_seleccionado]['Punto'].unique()
        opciones = [{'label': punto, 'value': punto} for punto in puntos_disponibles]
        return opciones
    else:
        return []

@app.callback(
    Output('hoja2', 'children'),
    Output('regresion-lineal-graph', 'figure'),
    Output('tabla-co2', 'data'),
    Output('fecha-selector', 'options'),
    Input('btn-regresion', 'n_clicks'),
    State('filtro-dia', 'value'),
    State('input-inicio-hora', 'value'),
    State('input-fin-hora', 'value'),
    State('filtro-punto', 'value') 
)
def aplicar_regresion_lineal(n_clicks, dia_seleccionado, inicio_hora, fin_hora, punto_seleccionado):
    global df2, data_count_df2, fechas_unicas
    fig_regresion_lineal = go.Figure()
    intervalo_co2_hora = pd.DataFrame()
    pendiente = 0 
    punto_etiqueta = None 
    latitud = None  
    longitud = None 
    options = []
    if not n_clicks or not inicio_hora or not fin_hora:
        return "Registros de Flujo CO2: 0", fig_regresion_lineal, [], options
    
    try:
        if n_clicks and inicio_hora and fin_hora:
            df_filtrado = df[df['Fecha'] == dia_seleccionado]
            inicio_tiempo = datetime.strptime(inicio_hora, '%H:%M:%S').time()
            fin_tiempo = datetime.strptime(fin_hora, '%H:%M:%S').time()
            
            if punto_seleccionado is not None:
                df_filtrado = df_filtrado[df_filtrado['Punto'] == punto_seleccionado]
                punto_etiqueta = df_filtrado['Punto'].iloc[0] 
                latitud = df_filtrado['Latitud'].iloc[0]  
                longitud = df_filtrado['Longitud'].iloc[0]  
            
            intervalo_co2_hora = df_filtrado[(df_filtrado['Fecha y Hora'].dt.time >= inicio_tiempo) & (df_filtrado['Fecha y Hora'].dt.time <= fin_tiempo)][['Hora', 'CO2']]
            intervalo_co2_hora['Recuento'] = range(1, len(intervalo_co2_hora) + 1)
            if len(intervalo_co2_hora) > 1:
                X = intervalo_co2_hora[['Recuento']]
                y = intervalo_co2_hora['CO2']
                    # Calcular R^2
                y_pred = regresion.predict(X)
                r_cuadrado = regresion.score(X, y)
                pendiente = regresion.coef_[0]
                intervalo_co2_hora['y_pred'] = y_pred
                intercepto = regresion.intercept_
                fig_regresion_lineal = create_regresion_lineal_graph(intervalo_co2_hora, pendiente, intercepto)
                
                if punto_seleccionado is not None:
                    # Enviar datos a la Hoja 2 de Google Sheets
                    fecha_seleccionada = datetime.strptime(dia_seleccionado, '%m/%d/%Y').strftime('%Y-%m-%d')
                    values = [[fecha_seleccionada, pendiente, punto_etiqueta, latitud, longitud,r_cuadrado]]
                    results = sheets.values().append(spreadsheetId=SPREADSHEETS_ID, range='Hoja 2!A1', 
                                                    valueInputOption='USER_ENTERED', 
                                                    body={'values': values}).execute()
                
                result2 = sheets.values().get(spreadsheetId=SPREADSHEETS_ID, range='Hoja 2!A:E').execute()
                values2 = result2.get('values', [])
                
                if values2:
                    expected_columns = ['Fecha Mes', 'Pendientes', 'Punto Medicion', 'Latitud', 'Longitud']
                    if values2[0] == expected_columns:
                        df2 = pd.DataFrame(values2[1:], columns=values2[0])
                        df2['Fecha Mes'] = pd.to_datetime(df2['Fecha Mes'])
                    else:
                        df2 = pd.DataFrame(columns=expected_columns)
                else:
                    df2 = pd.DataFrame(columns=['Fecha Mes', 'Pendientes', 'Punto Medicion', 'Latitud', 'Longitud'])
                    
        else:
            raise ValueError("Por favor, ingrese un formato de hora válido (HH:MM:SS)")
    except ValueError as e:
        raise dash.exceptions.PreventUpdate("Error de validación: " + str(e))
    fechas_unicas = df2['Fecha Mes'].dt.strftime('%m/%d/%Y').drop_duplicates().values
    options=[{'label': fecha, 'value': fecha} for fecha in fechas_unicas]
    data_count_df2 = len(df2)        
    return "Registros de Flujo CO2:"+f" {data_count_df2}",fig_regresion_lineal, intervalo_co2_hora.to_dict('records'),options



@app.callback(
    Output('grafico-co2-vs-distancia', 'figure'),
    Input('fecha-selector', 'value')
)
def co2vs_D(selected_dates):
    global df2

    if not selected_dates:
        # Si no se seleccionaron fechas, mostrar un mensaje o tomar alguna acción apropiada
        return create_empty_graph() 

    # Filtrar df2 por las fechas seleccionadas
    df2_cleaned = df2[df2['Fecha Mes'].isin(selected_dates)]

    if df2_cleaned.empty:
        # Si no hay datos para las fechas seleccionadas, mostrar un mensaje o tomar alguna acción apropiada
        return create_empty_graph()  

    df2_cleaned = df2_cleaned.dropna(subset=['Punto Medicion'])
    df2_cleaned = df2_cleaned[df2_cleaned['Punto Medicion'] != '']
    df2_cleaned['Punto Medicion'] = df2_cleaned['Punto Medicion'].astype(int)
    
    return create_flux_vs_distance_graph(df2_cleaned, selected_dates)


def create_empty_graph():
    fig = go.Figure()
    return fig

from datetime import datetime

def create_flux_vs_distance_graph(df2_cleaned, selected_dates):
    fig = go.Figure()
    print(selected_dates)
    
    
    all_puntos = sorted(df2_cleaned['Punto Medicion'].unique())
    
    for selected_date in selected_dates:
        # Convertir la fecha seleccionada a datetime.date
        selected_date = datetime.strptime(selected_date, '%m/%d/%Y').date()

        data = df2_cleaned[df2_cleaned['Fecha Mes'].dt.date == selected_date]
        
        pendientes_promedio = []
        
        for punto in all_puntos:
            punto_data = data[data['Punto Medicion'] == punto]
            
            if not punto_data.empty:
                # Si hay datos para el punto actual, calcula el promedio de pendientes
                pendientes_promedio.append(punto_data['Pendientes'].apply(lambda x: np.mean([float(val.strip()) for val in x.split()])).mean())
            else:
                # Si no hay datos para el punto actual, añade un cero
                pendientes_promedio.append(0)
        
        fig.add_trace(go.Scatter(x=all_puntos, y=pendientes_promedio, mode='markers+lines', name=f'Fecha {selected_date}'))
    
    fig.update_layout(
        title='CO2 vs Distancia', 
        font=dict(family='Manrope', size=10, color='black'),
        xaxis=dict(title='Punto Medicion', showgrid=True, tickvals=all_puntos, dtick=1),  
        yaxis=dict(title='Pendientes Promedio', showgrid=True, dtick=1), 
        plot_bgcolor='white',
        margin=dict(l=10, r=10, t=60, b=30),
    )
    fig.update_xaxes(gridcolor='lightgrey', linewidth=1)
    fig.update_yaxes(gridcolor='lightgrey', linewidth=1)
    return fig




if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 8050)))
