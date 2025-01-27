import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import os
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import warnings
warnings.filterwarnings("ignore")

class AutocorrelacaoTimeSeires(object):
    path_base = r"C:\Artigos Cientificos\Analise_Historica_MG\Files\Series Temporais\Autocorrelacao"

    dados_clima = [
        'precipitacao',
        'temperatura'
    ]

    def __init__(self):
        print('Obtendo Autocorrelação das Séries Temporais...')
        pass

    def control_time_series(self):
        data_climate, summary_climate = self.get_data_climate()
        data_climate_format = self.format_df_time_series(summary_climate)

        for index in self.dados_clima:
            print(index)

            self.plot_autocorrelacao(data_climate_format, index)
    
    def plot_autocorrelacao(self, df, index):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Autocorrelação da {index.title()} Mensal', fontsize=16, fontweight='bold')

        plot_acf(df[index], ax=axs[0], title='Autocorrelação Total')
        plot_pacf(df[index], ax=axs[1], title='Autocorrelação Parcial', method='ywm')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = os.path.join(self.path_base, f"autocorrelacao_{index}_autoregressive.png")
        plt.savefig(output_path)
        plt.clf()
    
    def format_df_time_series(self, df):
        df.mes= pd.to_datetime(df.mes)
        df.set_index('mes', inplace=True)

        print(df.info())
        print(df)

        return df
    
    def get_data_climate(self):
        sql_data = f'''
        SELECT
            ano,
            precipitacao,
            temperatura, 
            dt_registro
        FROM
            dados_mg.dados_metereologico
        WHERE
            ano::int NOT IN (2024)
        ORDER BY
            ano ASC
        '''

        conn = self.conn()
        cursor = conn.cursor()
        cursor.execute(sql_data)
        data = cursor.fetchall()

        columns = ['ano', 'precipitacao', 'temperatura', 'dt_registro']
        df = pd.DataFrame(data, columns=columns)

        df['dt_registro'] = pd.to_datetime(df['dt_registro'])
        df['mes'] = df['dt_registro'].dt.to_period('M').astype(str)

        summary_df = df.groupby('mes').agg(
            precipitacao=('precipitacao', 'mean'),
            temperatura=('temperatura', 'mean')
        ).reset_index()

        print(summary_df, '\n\n\n')

        return df, summary_df 
    
    def conn(self):
        conn = psycopg2.connect(
            host="localhost",
            database="artigo",
            user="postgres",
            password="vitor987sa@A",
            options="-c search_path=dados_mg"
        )

        return conn

def main():
    tsa = AutocorrelacaoTimeSeires()
    tsa.control_time_series()

if __name__ == '__main__':
    main()
    

    
