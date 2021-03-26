import pandas as pd
import numpy as np
from pathlib import Path
import re
import json

def normalizer(text, remove_tildes = True): #normalizes a given string to lowercase and changes all vowels to their base form
    text = text.lower() #string lowering
    text = re.sub(r'[^A-Za-zñáéíóú]', ' ', text) #replaces every punctuation with a space
    if remove_tildes:
        text = re.sub('á', 'a', text) #replaces special vowels to their base forms
        text = re.sub('é', 'e', text)
        text = re.sub('í', 'i', text)
        text = re.sub('ó', 'o', text)
        text = re.sub('ú', 'u', text)
    return text


def to_point(x):
    return x.replace(',','.')

def station_join(input_location = "data/raw/hrt/comunas_stations", output_location = "data/interim/metorological/hrt/comunas_stations/"):
    p = Path(input_location)
    subdirectories = [x for x in p.iterdir() if x.is_dir()]
    for dir_ in subdirectories:
        dir_aux = dir_
        str_replace = r'%s' % p
        dir_aux = r'%s' % dir_aux
        dir_aux = dir_aux.replace(str_replace, "")
        #comuna = re.sub('[^0-9a-zA-Z]+', '', dir_aux)
        comuna = re.sub(r'\\', '', dir_aux)
        sub_subd = [x for x in dir_.iterdir() if x.is_dir()]

        for sub_dir in sub_subd:
            dir_aux = sub_dir
            str_replace = r'%s' % dir_
            dir_aux = r'%s' % dir_aux
            dir_aux = dir_aux.replace(str_replace, "")
            station = re.sub(r'\\', '', dir_aux)
            csv_folder = sub_dir.rglob('*.csv')
            files = [x for x in csv_folder]
            dfs = [pd.read_csv(csv_file) for csv_file in files]
            df_aux = dfs.pop(0)
            for df in dfs:
                df_aux = pd.concat([df_aux, df]).reset_index(drop = True)    

            df_aux.to_csv( output_location +  comuna + "/" + comuna + "_" + station + ".csv", index = False)


def standarize(df, date_column = "Fecha Hora", format = '%d-%m-%Y %H:%M'):
    try:
        df.drop(columns = "Unnamed: 0", inplace = True)
    except:
        pass
    df.drop(columns = ["Presión atmosférica", "Veloc. máx. viento", "Dirección del viento", "Grados día (base 10)", "Horas frío (base 7)"])
    df[date_column] = pd.to_datetime(df[date_column], format = format)
    df["month"] = df[date_column].dt.month_name()
    df["month_number"] = df[date_column].dt.month
    df["day"] = df[date_column].dt.day_name()
    df["day_number"] = df[date_column].dt.day
    df["hour"] = df[date_column].dt.hour
    df["year"] = df[date_column].dt.year
  
    df = df[(df["hour"]<=20) & (df["hour"]>=5)].copy()

    ###### HAY QUE MEJORAR COMO LIDIAR INTERVALOS MAS PEQUEÑOS######
    df = df[df[date_column].dt.minute == 0].copy()

    numeric = list(set(df.columns).difference(set(["month",  "day",   date_column])))
    point_columns = ['Temp. promedio aire', 'Precipitación horaria',
        'Humed. rel. promedio', 'Radiación solar máx.',
        'Temp. Mínima', 'Temp. Máxima']
    df.loc[:, point_columns] = df.loc[:, point_columns].applymap(to_point)

    for var in numeric:
        df[var] = pd.to_numeric(df[var], errors = 'coerce')

    for col in  ['Humed. rel. promedio', 'Precipitación horaria', 'Radiación solar máx.']:
        negatives = df[df[col] < 0].index 
        for i in list(negatives):
            df.at[i, col] = np.nan
    #https://www.meteored.cl/tiempo-en_Curico-America+Sur-Chile-Maule--sactual-18574.html
    #Datos historicos de temperaturas extremas, se quitan si son mayores al menor o mayor historico(en los años)de la zona
    for col in  ['Temp. promedio aire',  'Temp. Mínima', 'Temp. Máxima']: 
        outliers = df[((df[col] > 37) | (df[col] < -6))].index 
        for i in list(outliers):
            df.at[i, col] = np.nan
    df.drop_duplicates(inplace = True)
    df.reset_index(drop = True, inplace = True )

    index_aux = (df[date_column].drop_duplicates()).index
    df = df.iloc[index_aux]
    df.reset_index(drop = True, inplace = True )

    return df

def get_nan_index(df):
    na_index = {}
    for col in list(df.columns):
        na_ind = np.where(df[col].isna())[0]
        na_index[col] = na_ind
        
    return na_index

def harversine_distance(lat1, lon1, lat2, lon2):
    r = 6371
    phi_1 = np.radians(lat1)
    phi_2 = np.radians(lat2)
    d_phi = np.radians(lat2-lat1)/2
    d_l = np.radians(lon2 - lon1)/2
    
    term1 = np.sin(d_phi)**2 + np.cos(phi_1)*np.cos(phi_2)*(np.sin(d_l)**2)
    term2 = r*(2*np.arctan2(np.sqrt(term1), np.sqrt(1-term1)))
    
    return np.round(term2, 8)

def nan_distance( df, row,  column, lat1, lon1,  dfs_dic, NA_INDEX , comuna_estaciones, station_column = "stations"):
    df_aux = comuna_estaciones.copy()
    df_aux['distancia'] =  harversine_distance(lat1, lon1, df_aux.Latitud, df_aux.Longitud)
    df_aux = df_aux.sort_values(by = 'distancia', ascending=True, inplace=False, na_position='last')
    df_aux.reset_index(inplace = True, drop = True)
    for i in df_aux.index: #recorro por distancia
        station = df_aux.iloc[i][station_column]
        nan_aux = NA_INDEX[station]

        year = df.iloc[row]["year"]
        month = df.iloc[row]["month_number"]
        day = df.iloc[row]["day_number"]
        hour = df.iloc[row]["hour"]
        station_aux = dfs_dic[station]
        condition = ((station_aux["year"] == year) &  (station_aux["month_number"] == month) & (station_aux["day_number"] == day) & (station_aux["hour"] == hour ))
    
        i_aux = station_aux[condition].index[0]
        value = station_aux[condition][column].values[0]
        if  i_aux not in nan_aux and not math.isnan(value):
            #value = station_aux[condition][column].values[0]
            return value

def nan_whitin(df, row, nan_index, column):
    min_column = ["Temp. Mínima"]
    max_column = ["Temp. Máxima", "Radiación solar máx."]
    avg_column = ["Temp. promedio aire", "Precipitación horaria"]
    if row!=(df.shape[0]-1):
        if row+1 not in nan_index:
            if row!=0:
                if row-1 not in nan_index:
                    if column in avg_column:
                        value = (1/2)*(df.iloc[row-1][column] + df.iloc[row+1][column]) 
                        return value
                    
                    elif column in max_column:
                        value = np.maximum(df.iloc[row-1][column], df.iloc[row+1][column])
                        return value
                        
                    elif column in min_column:
                        value = np.minimum(df.iloc[row-1][column], df.iloc[row+1][column])
                        return value
                        
                else: #row -1 esta en nan
                    value = df.iloc[row+1][column]
                    return value
                    
            else: #row == 0
                value = df.iloc[row+1][column]
                return value
                
        else: #row +1 esta en nan
            if row!=0:
                if row -1 not in nan_index:
                    value = df.iloc[row-1][column]
                    return value
                else:
                    value = -1
                    return value
                    
            else:
                value = -1
                return value
        #For good measure
        return -1

import math
def nan_filler(df, row, nan_index, column, lat1, lon1,  dfs_dic, NA_INDEX, comuna_estaciones, station_column = "stations"):
    value = nan_whitin(df, row, nan_index, column)
    if (value != -1) and not math.isnan(value):
        return value
    else:
        value = nan_distance(df, row,  column, lat1, lon1,  dfs_dic, NA_INDEX = NA_INDEX, comuna_estaciones = comuna_estaciones, station_column = station_column)
        return value

def get_comunas_midpoint(location = 'data/raw/hrt/chile_with_regions.json', region= 'Maule',
                        save = True, 
                        save_location ='data/processed/meteorological/hrt/comunas_maule.csv' ):
    #Descargado desde https://github.com/2x3-la/geo-chile
    f = open(location,) 
    chile = json.load(f) 
    comunas_maule = pd.DataFrame(chile[region])
    comunas_maule["name"]=comunas_maule.loc[:,["name"]].applymap(normalizer)
    comunas_maule["name"] = comunas_maule["name"].str.lower()
    for var in ["lat", "lng"]:
        comunas_maule[var] = pd.to_numeric(comunas_maule[var], errors = 'coerce')
    if save == True:
        comunas_maule.to_csv(save_location, index = False)
    return comunas_maule

class MeteorizerHrt():

    def __init__(self, join_raw_input = False, estaciones_comunas_maule_location = 'data/raw/hrt/estaciones_comunas_maule.csv'):
        if join_raw_input:
             station_join()
        self.comuna_estaciones = pd.read_csv(estaciones_comunas_maule_location)
        self.comunas_dic = None
        self.stations_dic = None
    
    def get_comuna_estaciones(self):
        return self.comuna_estaciones

    def add_station(self, latitud, longitud, comuna, stations):
        #comuna_estaciones tiene de base de la dataframe viene de https://climatologia.meteochile.gob.cl/application/informacion/buscadorDeEstaciones/
        #Luego se hizo un alcance con las estaciones que se possen y se agrega la columna "stations",  la cual contiene el nombre
        #de la estacion guardadae en data/interime/meteorological/hrt/comunas_stations
        comuna_estaciones = self.comuna_estaciones
        new_station = {"Latitud":latitud, "Longitud":longitud , "comuna": comuna, "stations": stations}
        df_aux = comuna_estaciones.copy()
        df_aux = df_aux.append(new_station, ignore_index = True)
        self.comuna_estaciones = df_aux

    def set_comunas_midpoint(self, location = 'data/raw/hrt/chile_with_regions.json', region= 'Maule'):
        self.comunas_maule = get_comunas_midpoint(location = location, region = region)
    
    def create_stations_dic(self, path = 'data/interim/metorological/hrt/comunas_stations'):
        p = Path(path)
        subdirectories = [x for x in p.iterdir() if x.is_dir()]
        stations_dic = {}
        for dir_ in subdirectories:
            csv_folder = dir_.rglob('*.csv')
            files = [x for x in csv_folder]
            file_names = []
            for file_name in files:
                str_replace = r'%s' % dir_
                file_aux = r'%s' % file_name
                file_aux = file_aux.replace(str_replace, "")
                file_aux = re.sub(r'\\', '', file_aux)
                file_aux = file_aux.replace(".csv", "")
                file_names.append(file_aux)
            # print(dir_, file_names)
            # print(" ")
            # print("-------------------------")
            dfs = [standarize(pd.read_csv(csv_file)) for csv_file in files]
            for (file_name, df) in list(zip(file_names, dfs)):
                stations_dic[file_name] = df
            #print(dfs)
            #stations_dic = {file_name:df for (file_name, df) in list(zip(file_names, dfs))}   

        self.stations_dic = stations_dic

    def create_comunas_dic(self, 
                        avg_columns = ['Temp. promedio aire','Humed. rel. promedio', 'Precipitación horaria'],
                        max_columns = ['Radiación solar máx.' ,'Temp. Máxima'], min_columns = ['Temp. Mínima']):
        
        comunas_dic  = {}
        comuna_estaciones = self.comuna_estaciones
        stations_dic = self.stations_dic
        for comuna in set(comuna_estaciones.comuna.values):
            aux = comuna_estaciones[comuna_estaciones["comuna"] == comuna]
            #Estaciones de la comuna:
            stations = aux["stations"].values
            station_0 = stations_dic[stations[0]]
            df_comuna = station_0.loc[:, ["Fecha Hora", "month", "month_number", "day","day_number", "hour", "year"]]
            cols = list(set(station_0.columns).difference(set(["Fecha Hora", "month", "month_number", "day","day_number", "hour", "year"])))
            for col in cols:
                stations_col = []
                for station_name in stations:
                    station = stations_dic[station_name]
                    stations_col.append(station[col])
                df_aux = pd.concat(stations_col, axis = 1)

                if col in min_columns:
                    new_col = df_aux.min(axis = 1)
                    df_comuna[col] = new_col

                elif col in max_columns:
                    new_col = df_aux.max(axis = 1)
                    df_comuna[col] = new_col
                elif col in avg_columns:
                    new_col = df_aux.mean(axis =1)
                    df_comuna[col] = new_col
    
            comunas_dic[comuna] = df_comuna

        self.comunas_dic = comunas_dic


    def create_na_dic(self, tipo ="comuna"):
        if tipo == "comuna":
            dfs_dic = self.comunas_dic
        elif tipo == "station":
            dfs_dic = self.stations_dic
        NA_INDEX = {}
        for df_name in dfs_dic:
            df = dfs_dic[df_name]
            na_index = get_nan_index(df)
            NA_INDEX[df_name] = na_index

        if tipo == "comuna":
            self.comuna_na_index = NA_INDEX
        elif tipo == "station":
            self.station_na_index = NA_INDEX

    def global_nan_filler(self, save_results = True,  df_comunas_location = 'data/interim/metorological/hrt/comunas/'):
        comuna_estaciones = self.comuna_estaciones
        df_comunas_dic = self.comunas_dic
        stations_dic = self.stations_dic
        comunas_na_index = self.comuna_na_index
        stations_na_index = self.station_na_index
        comunas_maule = self.comunas_maule
        new_dic = {}
        for comuna, df_comuna in df_comunas_dic.items():
            nan_index = comunas_na_index[comuna]
            columns = list(set(df_comuna.columns).difference(set(["Fecha Hora", "month", "month_number", "day","day_number", "hour", "year"])))
            lat1 = comunas_maule[comunas_maule["name"] == comuna].lat.values[0]
            lon1 = comunas_maule[comunas_maule["name"] == comuna].lng.values[0]
            df_comuna_copy = df_comuna.copy()
            for col in columns:
                for i in nan_index[col]:
                    value = nan_filler(df = df_comuna, row = 1, nan_index = nan_index, column = col,
                                        lat1 = lat1, lon1 = lon1, dfs_dic = stations_dic, NA_INDEX = stations_na_index,
                                        comuna_estaciones = comuna_estaciones )
                    df_comuna_copy.at[i,col] = value
            new_dic[comuna] = df_comuna_copy
        
        self.comunas_dic = new_dic
        if save_results:
            for (name, df) in new_dic.items():
                df.to_csv(df_comunas_location + name + ".csv", index = False)


#test = MeteorizerHrt(join_raw_input = False)
#test.set_comunas_midpoint()
#test.create_stations_dic()
#test.create_comunas_dic()
#test.create_na_dic("comuna")
#test.create_na_dic("station")
#test.global_nan_filler()