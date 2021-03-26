import pandas as pd
import numpy as np
from pathlib import Path
import re
def make_features(df, date_column = "Fecha Hora"):

    df["month"] = df["month"].astype('category')
    df["day"] = df["day"].astype('category')
    df["year"] = df["year"].astype('category')
    

    # Para crear features con los intervalos de 3 horas
    df0 = df[(df["hour"]>=8)&(df["hour"] <= 17)].copy()
    df0.reset_index(drop = True, inplace = True)
    
    df1 = df[(df["hour"]>=7)&(df["hour"] <= 16)].copy()
    df1.reset_index(drop = True, inplace = True)
    
    df2 = df[(df["hour"]>=9)&(df["hour"] <= 18)].copy()
    df2.reset_index(drop = True, inplace = True)
    
    #Base para agregar las features
    df_final = df0.loc[:, ["month_number", "day_number","month","day", "hour", "year"]]
    
    temp = "Temp. promedio aire"
    prec = "Precipitación horaria"
    hum = "Humed. rel. promedio"
    temp_m = "Temp. Mínima"
    temp_M = "Temp. Máxima"
    rad = "Radiación solar máx."
    
    
    df_final["Temperatura"] = df0[temp].values 
    #df_final["Temperatura_3h"] = (1/3)*(df0[temp].values + df1[temp].values + df2[temp].values)
    #df_final["Lluvia_3h"] = (1/3)*(df0[prec].values + df1[prec].values + df2[prec].values)
    df_final["Humedad"] = df0[hum].values
    #df_final["Humedad_3h"] = (1/3)*(df0[hum].values + df1[hum].values + df2[hum].values)
    df_final["Temperatura_min"] = df0[temp_m].values
    #df_final["Temperatura_min_3h"] = np.minimum(df0[temp_m].values, np.minimum(df1[temp_m].values, df2[temp_m].values))
    #df_final["Temperatura_max_3h"] = np.maximum(df0[temp_M].values, np.maximum(df1[temp_M].values, df2[temp_M].values))
    df_final["Temperatura_max"] = df0[temp_M].values
    #df_final["LLuvia_binaria_3h"] = np.where(df_final["Lluvia_3h"]!=0, 1, 0)
    df_final["LLuvia_binaria"] = np.where(df0[prec]!=0, 1, 0)
    #df_final["Radiacion_solar_3h"] =  np.maximum(df0[rad].values, np.maximum(df1[rad].values, df2[rad].values))
    df_final["Radiacion_solar"] =  df0[rad].values
    #df_final.drop(columns = ["Lluvia_3h"], inplace = True)
    
    #df_final["hour"] = df_final["hour"].astype('category')

    return df_final

class FeaturizerHrtMeteoro:
   def __init__(self, comuna_dic_location = 'data/interim/metorological/hrt/comunas',
               save_output = True,
               output_location = 'data/processed/meteorological/hrt/comunas_features/'):
      self.features_location = output_location
      txt_folder = Path(comuna_dic_location).rglob('*.csv')
      files = [x for x in txt_folder]
      dfs = [make_features(pd.read_csv(csv_file)) for csv_file in files]
      file_names = []
      for file_name in files:
         file_aux = r'%s' % file_name
         file_aux = re.sub(r'.*\\', '', file_aux)
         file_aux = file_aux.replace(".csv", "")
         file_names.append(file_aux)
      comunas_dic = {file_name:df for (file_name, df) in list(zip(file_names, dfs))}
      self.comunas_dic = comunas_dic
      if save_output:
         for (file_name, df) in comunas_dic.items():
            df.to_csv(output_location + file_name +'.csv', index = False)

test = FeaturizerHrtMeteoro()
#print(test.comunas_dic["teno"])