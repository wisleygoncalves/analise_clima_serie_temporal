import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import os
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr, anderson, levene, kruskal, shapiro, norm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf, q_stat 
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg

import warnings
warnings.filterwarnings("ignore")

class TestTrainARIMA(object):
    path_base = r"C:\Artigos Cientificos\Analise_Historica_MG\Files\Series Temporais\TestBestTimesSeries"

    dados_clima = [
        'precipitacao',
        'temperatura'
    ]

    def __init__(self):
        print('Testando Melhor Modelo de Série Temporal...')
        pass
    
    def control_time_series(self):
        data_climate, summary_climate = self.get_data_climate()
        data_climate_format = self.format_df_time_series(summary_climate)

        for index in self.dados_clima:
            print(index)

            list_predict_arima, train_set_arima, test_set_arima = self.train_test_arima(data_climate_format, index)
            list_predict_autoregressive, train_set, test_set = self.train_test_autoregressive(data_climate_format, index)

            list_predict = list_predict_autoregressive + list_predict_arima 

            self.diagram_correlation(list_predict, test_set, index)
    
    
    def diagram_correlation(self, list_predict, test_set, index):
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f'Correlação dos Resíduos dos Dados de {index.title()}', fontsize=16, fontweight='bold')

        axs_flat = axs.flat

        for ax, predict in zip(axs_flat, list_predict):
            ordem, previsao, aic, bic, residuos = predict

            previsao = previsao.fillna(0)
            test_set = test_set.fillna(0)

            x = test_set
            y = previsao

            correlacao, _ = pearsonr(x, y)

            #coef = np.polyfit(x, y, 1)
            #reta = np.poly1d(coef)

            ax.scatter(x, y, label='Dados', color='gray')
            #ax.plot(x, reta(x), color='red', label=f'Reta de Tendência (y = {coef[0]:.2f}x + {coef[1]:.2f})')

            dw_stat = durbin_watson(residuos)
            ax.annotate(f'DW: {dw_stat:.2f}', xy=(0.65, 0.95), xycoords='axes fraction', fontsize=10, 
                        ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='white', facecolor='white'))

            lags = min(10, len(residuos) // 5)  
            acf_values = acf(residuos, nlags=lags, fft=False)  
            q_stat_values, p_values = q_stat(acf_values[1:], len(residuos))  

            ax.annotate(f'Ljung-Box p-valor: {p_values[-1]:.4f}', xy=(0.65, 0.85), xycoords='axes fraction', 
                        fontsize=10, ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='white', facecolor='white'))

            plot_acf(residuos, ax=ax, lags=lags, alpha=0.05, title=f'{ordem} - Pearson: {correlacao:.2f}')

            ax.set_title(f'{ordem} - Correlação Pearson: {correlacao:.2f}', fontweight='bold') 
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=10)

            ax.grid(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = os.path.join(self.path_base, f"correlacao_{index}_autoregressive.png")
        plt.savefig(output_path)


    def train_test_arima(self, df, index):
        list_predict = []

        print(index)
        result = adfuller(df[index].dropna())
        print("A série é estacionária" if result[1] < 0.05 else "A série NÃO é estacionária")

        if result[1] > 0.05:
            print(index)
            df[index] = df[index].diff().dropna()

            result = adfuller(df[index].dropna())
            print("A série é estacionária" if result[1] < 0.05 else "A série NÃO é estacionária")

            if result[1] > 0.05:
                quit()

        train_size = int(len(df[index]) * 2/3)
        #print(f'Train: {train_size}')

        train_set = df[index][:train_size].astype(float)
        test_set = df[index][train_size:].astype(float)

        train_set = train_set.fillna(0)
        test_set = test_set.fillna(0)

        i = None

        if index == 'precipitacao':
            i = 9
        
        if index == 'temperatura':
            i = 11

        arma_train = ARIMA(train_set, order=(i, 0, i)).fit()
        arma_test = ARIMA(test_set, order=(i, 0, i)).fit(arma_train.params)

        predict = arma_test.predict()

        residuos = test_set - predict
        
        list_predict.append((
            f'ARMA ({i}, 0, {i})',
            predict,
            arma_train.aic,
            arma_train.bic,
            residuos.dropna()
        ))

        print(list_predict)


        return list_predict, train_set, test_set


    def train_test_autoregressive(self, df, index):
        list_predict = []
        
        print(index)
        result = adfuller(df[index].dropna())
        print("A série é estacionária" if result[1] < 0.05 else "A série NÃO é estacionária")

        if result[1] > 0.05:
            print(index)
            df[index] = df[index].diff().dropna()

            result = adfuller(df[index].dropna())
            print("A série é estacionária" if result[1] < 0.05 else "A série NÃO é estacionária")

            if result[1] > 0.05:
                quit()
            
           
        train_size = int(len(df[index]) * 2/3)
        #print(f'Train: {train_size}')

        train_set = df[index][:train_size]
        test_set = df[index][train_size:]

        train_set = train_set.fillna(0)
        test_set = test_set.fillna(0)

        lags = None

        if index == 'precipitacao':
            lags = 5
        
        if index == 'temperatura':
            lags = 9

        ar_model = AutoReg(train_set.to_numpy(), lags=lags, trend='c').fit()
        params = ar_model.params
        predict = params[0]

        for i, coef in enumerate(ar_model.params[1:]):
            predict += coef * test_set.shift(i + 1)
        
        residuos = test_set - predict
        
        list_predict.append((
            f'AR {lags}',
            predict,
            ar_model.aic,
            ar_model.bic,
            residuos.dropna()
        ))

        print(list_predict)

        return list_predict, train_set, test_set


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
    tsa = TestTrainARIMA()
    tsa.control_time_series()

if __name__ == '__main__':
    main()
