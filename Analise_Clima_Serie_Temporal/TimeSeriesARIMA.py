import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
import numpy as np

import warnings
warnings.filterwarnings("ignore")


class TimeSeriesARIMA(object):
    path_base = r"C:\Artigos Cientificos\Analise_Historica_MG\Files\Series Temporais\ARIMA"

    dados_clima = [
        'precipitacao',
        'temperatura'
    ]

    def __init__(self):
        print('Obtendo Séries Temporais ARIMA...')
        pass

    def control_time_series(self):
        data_climate, summary_climate = self.get_data_climate()
        data_climate_format = self.format_df_time_series(summary_climate)

        for index in self.dados_clima:
            print(index)

            list_predict, list_pvalues, train_set, test_set, list_residuals = self.train_test_arima(data_climate_format, index)
            
            self.plot_train_test_arima(list_predict, train_set, test_set, index)
            self.plot_histogram_pvalues(list_pvalues, index)
            self.plot_residuals_regression(list_residuals, index)
            self.plot_best_model(data_climate_format, index)
    
    def plot_train_test_arima(self, list_predict, train_set, test_set, index):
        #fig, axs = plt.subplots(6, 1, figsize=(14, 10))
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f'Train Test da {index.title()} Mensal - ARIMA', fontsize=16, fontweight='bold')

        axs_flat = axs.flat

        for ax, predict in zip(axs_flat, list_predict):
            ordem, eqm, previsao = predict 

            ax.plot(train_set, linestyle='-', color='black', label='Treino')  
            ax.plot(test_set, linestyle='--', color='red', label='Teste')  
            ax.plot(previsao, linestyle='-', color='green', label=f'Previsão (EQM={eqm})')

            ax.set_title(ordem, fontweight='bold') 
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=10)
            ax.grid(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = os.path.join(self.path_base, f"train_test_{index}_arima.png")
        plt.savefig(output_path)

    def plot_histogram_pvalues(self, list_pvalues, index):
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f'Distribuição dos p-valores - {index.title()}', fontsize=16, fontweight='bold')

        axs_flat = axs.flat

        for ax, (label, pvalues) in zip(axs_flat, list_pvalues):
            bins = np.linspace(0, 1, 15)
            ax.hist(pvalues, bins=bins, color='lightgray', edgecolor='black', alpha=0.7, density=True, label='p-valores')

            mu, sigma = norm.fit(pvalues)
            x = np.linspace(0, 1, 100)
            pdf = norm.pdf(x, mu, sigma)
            ax.plot(x, pdf, color='red', lw=2, label=f'Normal (μ={mu:.2f}, σ={sigma:.2f})')

            ax.set_title(label, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_xlabel('p-valores')
            ax.set_ylabel('Densidade')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_path = os.path.join(self.path_base, f"histograma_pvalues_{index}_arima.png")
        plt.savefig(output_path)
        plt.clf()
    
    def plot_residuals_regression(self, list_residuals, index):
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f'ARIMA Resíduos - {index.title()}', fontsize=16, fontweight='bold')

        axs_flat = axs.flat

        for ax, (label, residuals) in zip(axs_flat, list_residuals):
            x = np.arange(len(residuals)).reshape(-1, 1)
            y = residuals

            model = LinearRegression()
            model.fit(x, y)
            y_pred = model.predict(x)

            eqm = mean_squared_error(y, y_pred)

            ax.scatter(x, y, color='blue', alpha=0.5, label='Resíduos')
            ax.plot(x, y_pred, color='red', linewidth=2, label=f'Regressão Linear (EQM={eqm:.4f})')

            ax.set_title(label, fontweight='bold')
            ax.set_xlabel('Índice')
            ax.set_ylabel('Resíduo')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
            ax.grid(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_path = os.path.join(self.path_base, f"regressao_residuos_{index}_arima.png")
        plt.savefig(output_path)
        plt.clf()
    
    def plot_best_model(self, df, index):
        arima_model = None

        data = df[index].astype(float).dropna()

        if 'precipitacao' in index:
            arima_model = ARIMA(data, order=(12, 0, 12)).fit()
        
        if 'temperatura' in index:
            arima_model = ARIMA(data, order=(12, 0, 12)).fit()
        
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.suptitle(f'Série Temporal ARIMA - {index.title()}', fontsize=16, fontweight='bold')

        ax.plot(data, linestyle='--', color='black', label='Real', linewidth=2)  
        ax.plot(arima_model.predict(), linestyle='-', color='red', label='Previsão', linewidth=2.5)

        ax.legend(loc='upper center', ncol=2, fontsize=10)
        ax.grid(False)

        output_path = os.path.join(self.path_base, f"serie_temporal_{index}_arima.png")
        plt.savefig(output_path)
        plt.clf()

    def train_test_arima(self, df, index):
        list_predict = []
        list_pvalues = []
        list_residuals = []

        train_size = int(len(df[index]) * 2/3)
        print(f'Train: {train_size}')

        train_set = df[index][:train_size].astype(float).dropna()
        test_set = df[index][train_size:].astype(float).dropna()

        print(train_set, '\n\n')
        print(test_set)

        for i in range(2, 13, 2):
            arma_train = ARIMA(train_set, order=(i, 0, i)).fit()
            arma_test = ARIMA(test_set, order=(i, 0, i)).fit(arma_train.params)

            predict = arma_test.predict()

            residuals = arma_test.resid

            p_values = arma_test.pvalues

            eqm = round((residuals ** 2).mean(), 4)
            
            list_predict.append((
                f'Ordem ({i}, 0, {i})',
                eqm,
                predict
            ))

            print(list_predict)

            list_pvalues.append((
                f'Ordem ({i}, 0, {i})',
                p_values
            ))

            print(list_pvalues)
            
            list_residuals.append((
                f'Ordem ({i}, 0, {i})',
                residuals
            ))

            print(list_residuals)

        return list_predict, list_pvalues, train_set, test_set, list_residuals

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
    tsa = TimeSeriesARIMA()
    tsa.control_time_series()

if __name__ == '__main__':
    main()