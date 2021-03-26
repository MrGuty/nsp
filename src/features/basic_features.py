import pandas as pd
import sklearn.preprocessing
import logging
import numpy as np
import sqlite3
from io import StringIO
from pathlib import Path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_historic_p(db_location,citation_date_column,grouping_columns,nsp_column,nsp_p_column_name):
    history = pd.read_sql_table("history", f'sqlite:///{db_location}').sort_values(by=citation_date_column)
    history["citations_cumcount"] = history.groupby(by=grouping_columns)[nsp_column].cumcount()
    history["nsp_cumsum"] = history.groupby(by=grouping_columns)[nsp_column].cumsum()
    history[nsp_p_column_name] = history["nsp_cumsum"] / history["citations_cumcount"]
    history[nsp_p_column_name] = np.where(history["citations_cumcount"] == 0,0.5,history[nsp_p_column_name])
    del history["citations_cumcount"]
    del history["nsp_cumsum"]
    return history

######################
##########NUEVO########
#######################
def harversine_distance(lat1, lon1, lat2, lon2):
    r = 6371
    phi_1 = np.radians(lat1)
    phi_2 = np.radians(lat2)
    d_phi = np.radians(lat2-lat1)/2
    d_l = np.radians(lon2 - lon1)/2
    
    term1 = np.sin(d_phi)**2 + np.cos(phi_1)*np.cos(phi_2)*(np.sin(d_l)**2)
    term2 = r*(2*np.arctan2(np.sqrt(term1), np.sqrt(1-term1)))
    
    return np.round(term2, 8)

def asigna_comunas(comunas, comuna_est_location = 'data/raw/hrt/estaciones_comunas_maule.csv', 
                    comuna_region_location = 'data/processed/meteorological/hrt/comunas_maule.csv',
                    save_result = True,
                    output_location = 'data/processed/meteorological/hrt/comuna_asignada.csv' ):
  comuna_est = pd.read_csv(comuna_est_location)
  comuna_region = pd.read_csv(comuna_region_location)
  comuna_asignada = []
  for i in range(comunas.shape[0]):
    com = comunas.iloc[i][0]
    comuna_aux = comuna_region[comuna_region['name'] == com]
    if (com not in list(comuna_region.name)) or (comuna_aux.shape[0] == 0):
      comuna_asignada.append("pencahue")
    elif com in comuna_est:
      comuna_asignada.append(com) 
    else:
      lat1 = float(comuna_aux.lat.values[0])
      lon1 = float(comuna_aux.lng.values[0])
      comuna_region_aux = comuna_region.copy()
      comuna_region_aux['distancia'] = harversine_distance(lat1, lon1, comuna_region_aux.lat , comuna_region_aux.lon)
      comuna_region_aux = comuna_region_aux.sort_values(by = 'distancia', ascending=True, inplace=False, na_position='last')
      for com2 in comuna_region_aux["name"].values:
        if com2 in comuna_est:
          comuna_asignada.append(com2)

    if save_result:
        df_aux = pd.DataFrame(comuna_asignada)
        df_aux.to_csv(output_location, index = False)

  return comuna_asignada


def merge_features(data, col_comuna = "comuna_asignada", dfs_dic):
    df_aux_dic = {}
    data = data.drop_duplicates()
    for comuna in list(set(data[col_comuna].values)):
        #print(comuna)
        features_comuna = dfs_dic[comuna]
        condition = (data[col_comuna] == comuna)
        df_comuna = data.loc[condition].copy()
        df_aux_dic[comuna] = df_comuna.merge( features_comuna,how = "left", 
                                     left_on = ["month_number", "month", "day_number", "day", "hour", "year"],  
                                     right_on =["month_number", "month", "day_number", "day", "hour", "year"] )
        aux_comuna = comuna
        df_final = pd.DataFrame(columns = df_aux_dic[aux_comuna].columns)
    for df in df_aux_dic.values():
        df_final = df_final.append(df, ignore_index = True)
    df_final.reset_index(drop = True, inplace = True)
    return df_final

class Featurizer:
    def __init__(self,interim_dataset_location):
        self.data = pd.read_csv(interim_dataset_location)
        logger.info("current shape: {}".format(self.data.shape))
    def generate_basic_features(self,use_reserve=True, idx="PAID",db_location="data/processed/history2.sqlite"):
        self.idx = idx
        self.data["FechaCita"] = pd.to_datetime(self.data["FechaCita"])
        self.data["HoraCita"] = pd.to_datetime(self.data["HoraCita"])
        self.data["FechaNac"] = pd.to_datetime(self.data["FechaNac"])
        if use_reserve:
            self.data["FechaReserva"] = pd.to_datetime(self.data["FechaReserva"])
            self.data["delay"] = (self.data["FechaCita"]-self.data["FechaReserva"]).astype('timedelta64[W]')
        
        self.historic_p_s = get_historic_p(db_location,"FechaCita",[self.idx,"Especialidad"],"NSP","nsp_p")
        self.historic_p_g = get_historic_p(db_location,"FechaCita",[self.idx],"NSP","nsp_p_g")
        self.data = self.data.merge(self.historic_p_s,how="left")
        self.data = self.data.merge(self.historic_p_g,how="left")
        
        self.data["age"] = (self.data["FechaCita"]-self.data["FechaNac"]).astype('timedelta64[D]')/365.25
        self.data["month"] = self.data["FechaCita"].dt.month_name()
        self.data["day"] = self.data["FechaCita"].dt.day_name()
        self.data["hour"] = self.data["HoraCita"].dt.hour

        self.data = self.data[self.data["EstadoCita"].isin(["Atendido","No Atendido"])]
        self.data["NSP"] = np.where(self.data["EstadoCita"] == "No Atendido",1,0)

        
        logger.info("current shape: {}".format(self.data.shape))

        self.data = self.data[(self.data["age"] <= 15) & (self.data["age"] >= 0)]
        if use_reserve:
            self.data = self.data[(self.data["delay"] <= 10) & (self.data["delay"] >= 0)]
        self.data = self.data[(self.data["hour"] <= 17) & (self.data["hour"] >= 8)]
        self.data["hour"] = self.data["hour"].astype('category')
        if use_reserve:
            self.data["delay"] = self.data["delay"].astype('category')
        logger.info("current shape: {}".format(self.data.shape))

        self.data["age"] = pd.cut(self.data["age"],[0,0.5,5,12,18],right=False,labels=["lactante","infante_1","infante_2","adolescente"])
        self.data_to_history = self.data[[idx,"Especialidad","FechaCita"]]

        ## Crear clasificacion TipoPrestacion medica/no medica a partir de los codigos FONASA

        # Lista de codigos de consultas medicas
        CodigosConsultaMedica = ['101111','101112','101113','903001']

        # Lista de codigos de consultas no medicas
        CodigosConsultaNoMedica = ['903002','903303','102001','102005','102006','102007',
                                '1101004','1101011','1101041','1101043','1101045','1201009',
                                '1301008','1301009']

        # Lista de codigos de procedimientos
        CodigosProcedimiento = ['305048','901005','1101009','1101010','1701003',
                                '1701006','1701045','1707002','1901030','2701013',
                                '2701015','2702001','AA09A3','AA09A4','Estudio']

        self.data['TipoPrestacionC'] = 'OTRO'

        self.data.loc[self.data['CodPrestacion'].isin(CodigosConsultaMedica),'TipoPrestacionC'] = 'ConsultaMedica'
        self.data.loc[self.data['CodPrestacion'].isin(CodigosConsultaNoMedica),'TipoPrestacionC'] = 'ConsultaNoMedica'
        self.data.loc[self.data['CodPrestacion'].isin(CodigosProcedimiento),'TipoPrestacionC'] = 'Procedimiento'
        #DD.loc[np.logical_not(DD['CodPrestacion'].isin(CodigosConsultaMedica+CodigosConsultaNoMedica+CodigosProcedimiento)),'TipoPrestacionC'] = 'OTRO'

        ## Crear clasificacion Profesional Medico/No medico

        # lista de tipos de profesionales medicos
        Profesional_medico = ['Médico','Médico Cirujano','Odontólogo/Dentista',
                            'Cirujano(a) Dentista','Ginecólogo(a)','Psiquiatra']

        #map(unicode,Profesional_medico)

        # lista de tipos de profesionales no medicos
        Profesional_noMedico = ['Enfermera (o)','Psicólogo (a)',#'No Mencionada',
                                'Kinesiólogo (a)','Fonoaudiólogo (a)','Tecnólogo Médico',
                                'Nutricionista','Terapeuta Ocupacional','Asistente Social',
                                'Técnico Paramédico']

        #map(unicode,Profesional_medico)

        # crear columna TipoProfesionalC (Clasificacion)
        self.data['TipoProfesionalC'] = 'OTRO'

        # transformar las entradas de TipoProfesional unicode a string
        self.data['TipoProfesional'].apply(lambda x: str(x))

        # decir si el tipo de profesional es medico o no medico, guardar en TipoProfesionalC
        self.data.loc[self.data['TipoProfesional'].astype(str).isin(Profesional_medico),'TipoProfesionalC'] = 'Medico'
        self.data.loc[self.data['TipoProfesional'].astype(str).isin(Profesional_noMedico),'TipoProfesionalC'] = 'NoMedico'
        #DD.loc[np.logical_not(DD['TipoProfesional'].isin(Profesional_medico)),'TipoProfesionalC'] = 'NoMedico'

        #print DD.loc[DD['TipoProfesionalC']=='Medico']['TipoProfesional'].value_counts()
        #print ''
        #
        #print DD.loc[DD['TipoProfesionalC']=='NoMedico']['TipoProfesional'].value_counts()
        if use_reserve:
            self.data = self.data.drop(columns=[idx,"FechaCita","HoraCita","FechaNac","FechaReserva","EstadoCita",'TipoProfesional', 'CodPrestacion'], axis=1)
        else:
            self.data = self.data.drop(columns=[idx,"FechaCita","HoraCita","FechaNac","EstadoCita",'TipoProfesional', 'CodPrestacion'], axis=1)
        logger.info(self.data.columns)
        self.data = pd.get_dummies(self.data)
        logger.info("current shape: {}".format(self.data.shape))
        
    def write(self,data_location):
        self.data.dropna(inplace=True)
        logger.info(self.data.columns)
        self.data.to_csv(data_location,index=False)


class FeaturizerCrsco:
    def __init__(self,interim_dataset_location):
        self.data = pd.read_csv(interim_dataset_location)
        logger.info("current shape: {}".format(self.data.shape))
    def generate_basic_features(self,idx="Rut",db_location="data/processed/crsco_history.sqlite"):
        self.data["FechaCita"] = pd.to_datetime(self.data["FechaCita"])
        self.data.sort_values(by="FechaCita",ascending=True,inplace=True)
        self.data["HoraCita"] = pd.to_datetime(self.data["HoraCita"])
        self.data["FechadeNac"] = pd.to_datetime(self.data["FechadeNac"])
        
        self.historic_p_s = get_historic_p(db_location,"FechaCita",[idx,"Especialidad"],"NSP","nsp_p")
        self.historic_p_g = get_historic_p(db_location,"FechaCita",[idx],"NSP","nsp_p_g")
        self.data = self.data.merge(self.historic_p_s,how="left")
        self.data = self.data.merge(self.historic_p_g,how="left")

        self.data["age"] = (self.data["FechaCita"]-self.data["FechadeNac"]).astype('timedelta64[D]')/365.25
        self.data["month"] = self.data["FechaCita"].dt.month_name()
        self.data["day"] = self.data["FechaCita"].dt.day_name()
        self.data["hour"] = self.data["HoraCita"].dt.hour

        self.data = self.data[self.data["EstadoCita"].isin(["Atendido","No Atendido"])]
        self.data["NSP"] = np.where(self.data["EstadoCita"] == "No Atendido",1,0)

        
        logger.info("current shape: {}".format(self.data.shape))

        self.data = self.data[(self.data["hour"] <= 17) & (self.data["hour"] >= 8)]
        self.data["hour"] = self.data["hour"].astype('category')
        logger.info("current shape: {}".format(self.data.shape))

        self.data["age"] = pd.cut(self.data["age"],[0,0.5,5,12,18,26,59,100],right=False,include_lowest=True,labels=["lactante","infante_1","infante_2","adolescente","joven","adulto","adulto mayor"])

        self.data = self.data.drop(columns=["Rut","FechaCita","HoraCita","FechadeNac","EstadoCita", 'CodigoPrestacion'], axis=1)
        logger.info(self.data.columns)
        self.data = pd.get_dummies(self.data)
        logger.info("current shape: {}".format(self.data.shape))

    def write(self,data_location):
        self.data.dropna(inplace=True)
        logger.info(self.data.columns)
        self.data.to_csv(data_location,index=False)


class FeaturizerHrt:
    def __init__(self,interim_dataset_location, 
                add_comuna = False, comuna_location = 'data/processed/meteorological/hrt/comuna_asignada.csv',
                add_comunas_dic = False,
                comunas_location = 'data/processed/meteorological/hrt/comunas_features/'):
        self.data = pd.read_csv(interim_dataset_location)
        if add_comuna:
            comuna_asignada = pd.read_csv(comuna_location)
            self.data["comuna_asignada"] = comuna_asignada
        if add_comunas_dic:
            txt_folder = Path('Dataset/hrt/comunas_stations_features').rglob('*.csv')
            files = [x for x in txt_folder]
            dfs = [pd.read_csv(csv_file) for csv_file in files]
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
        logger.info("current shape: {}".format(self.data.shape))
    def generate_basic_features(self,idx="RUT",db_location="data/processed/hrt_history.sqlite", meteoro = False):
        self.data["FECHA_CITA"] = pd.to_datetime(self.data["FECHA_CITA"])
        self.data.sort_values(by="FECHA_CITA",ascending=True,inplace=True)
        self.data["HORA_CITA"] = pd.to_datetime(self.data["HORA_CITA"])
        self.data["FECHANAC"] = pd.to_datetime(self.data["FECHANAC"])

        self.historic_p_s = get_historic_p(db_location,"FECHA_CITA",[idx,"ESPECIALIDAD"],"NSP","nsp_p")
        self.historic_p_g = get_historic_p(db_location,"FECHA_CITA",[idx],"NSP","nsp_p_g")
        self.data = self.data.merge(self.historic_p_s,how="left")
        self.data = self.data.merge(self.historic_p_g,how="left")

        self.data["FECHA_RESERVA"] = pd.to_datetime(self.data["FECHA_RESERVA"])
        self.data["delay"] = (self.data["FECHA_CITA"]-self.data["FECHA_RESERVA"]).astype('timedelta64[W]')
        self.data["age"] = (self.data["FECHA_CITA"]-self.data["FECHANAC"]).astype('timedelta64[D]')/365.25
        self.data["month"] = self.data["FECHA_CITA"].dt.month_name()
        self.data["day"] = self.data["FECHA_CITA"].dt.day_name()
        self.data["hour"] = self.data["HORA_CITA"].dt.hour

        self.data["NSP"] = np.where(self.data["FECHA_HORA_CONFIRMACION_CITA"].isna(),1,0)

        
        logger.info("current shape: {}".format(self.data.shape))

        self.data = self.data[(self.data["hour"] <= 17) & (self.data["hour"] >= 8)]
        self.data = self.data[(self.data["delay"] <= 10) & (self.data["delay"] >= 0)]
        self.data["delay"] = self.data["delay"].astype('category')

        ######NUEVO##########
        #Necesario para merge con las nuevas features
        if meteoro:
            self.data["month_number"] = self.data["FECHA_CITA"].dt.month
            self.data["day_number"] = self.data["FECHA_CITA"].dt.day
            self.data["year"] = self.data["FECHA_CITA"].dt.year
            self.data = merge_features(self.data, dfs_dic = self.comunas_dic)
            self.data.drop(inplace = True, columns = ["comuna_asignada", "year", "day_number", "month_number"])
            self.data[ 'LLuvia_binaria'] = pd.to_numeric(self.data[ 'LLuvia_binaria'])
            self.data["Temperatura"] = discretize(self.data.loc[:,["Temperatura"]], bins = 7, strategy='uniform')
            self.data["Temperatura_min"] = discretize(self.data.loc[:,["Temperatura_min"]], bins = 8, strategy='uniform')
            self.data["Temperatura_max"] = discretize(self.data.loc[:,["Temperatura_max"]], bins = 7, strategy='uniform')
            self.data["Humedad"] = discretize(self.data.loc[:,["Humedad"]], bins = 7, strategy='kmeans')
            self.data["Radiacion_solar"] = discretize(self.data.loc[:,["Radiacion_solar"]], bins = 38, strategy='kmeans')
        
            feature_columns = ["Temperatura", 'Humedad','Temperatura_min', 'Temperatura_max', 'Radiacion_solar']

            for feature in feature_columns:
                self.data[feature] = self.data[feature].astype('category')
        #####################

        self.data["hour"] = self.data["hour"].astype('category')
        logger.info("current shape: {}".format(self.data.shape))

        self.data["age"] = pd.cut(self.data["age"],[0,0.5,5,12,18,26,59,100],right=False,include_lowest=True,labels=["lactante","infante_1","infante_2","adolescente","joven","adulto","adulto mayor"])
        self.data_to_history = self.data[["RUT","ESPECIALIDAD","FECHA_CITA"]]

        self.data = self.data.drop(columns=["RUT","FECHA_CITA","HORA_CITA","FECHANAC","FECHA_HORA_CONFIRMACION_CITA","FECHA_RESERVA"], axis=1)
        logger.info(self.data.columns)
        self.data = pd.get_dummies(self.data)
        logger.info("current shape: {}".format(self.data.shape))

    def write(self,data_location):
        self.data.dropna(inplace=True)
        logger.info(self.data.columns)
        self.data.to_csv(data_location,index=False)