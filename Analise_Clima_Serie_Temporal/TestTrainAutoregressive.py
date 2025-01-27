import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr, levene, kruskal, shapiro, norm
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf, q_stat 
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

class TestTrainAutoregressive(object):
    path_base = r"C:\Artigos Cientificos\Analise_Historica_MG\Files\Series Temporais\TestTrainAR"

    dados_clima = [
        'precipitacao',
        'temperatura'
    ]

    def __init__(self):
        print('Obtendo Séries Temporais Autoregressive...')
        pass


    def control_time_series(self):
        data_climate, summary_climate = self.get_data_climate()
        data_climate_format = self.format_df_time_series(summary_climate)

        for index in self.dados_clima:
            print(index)

            list_predict, train_set, test_set = self.train_test_autoregressive(data_climate_format, index)

            self.plot_train_test_autoregressive(list_predict, train_set, test_set, index)
            self.calcute_metrics_autoregressive(list_predict, test_set, index)
            list_test_norm_data, list_test_homoge = self.test_data_times_series(list_predict, index)
            self.plot_curve_norm(list_predict, list_test_norm_data, index)
            self.test_kruskal_wallis_times_series(list_predict, index)
            self.diagram_correlation(list_predict, test_set, index)


    def plot_train_test_autoregressive(self, list_predict, train_set, test_set, index):
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f'Train Test da {index.title()} Mensal - Autoregressive', fontsize=16, fontweight='bold')

        axs_flat = axs.flat

        for ax, predict in zip(axs_flat, list_predict):
            ordem, previsao, aic, bic, residuos = predict 

            ax.plot(train_set, linestyle='-', color='black', label='Treino')  
            ax.plot(test_set, linestyle='--', color='red', label='Teste')  
            ax.plot(previsao, linestyle='-', color='green', label=f'Previsão')

            ax.set_title(ordem, fontweight='bold') 
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=10)
            ax.grid(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = os.path.join(self.path_base, f"train_test_{index}_autoregressive.png")
        plt.savefig(output_path)
    

    def calcute_metrics_autoregressive(self, list_predict, test_set, index):
        predict_metrics = []
        list_metrics_predict = []

        for data_predict in list_predict:
            predict_metrics.clear()

            predict = data_predict[1]
            residuos = data_predict[4]

            predict_filter = predict.fillna(0)

            eqm = mean_squared_error(test_set, predict_filter)

            list_metrics_predict.append({
                'eqm': round(eqm, 3),
                'rmse': round(np.sqrt(eqm), 3),
                'mape': round((np.mean(np.abs(residuos / test_set)) * 100), 3),
                'dp': round(np.std(predict_filter), 2),
                'r2': round(r2_score(test_set, predict_filter), 2),
                'aic': round(data_predict[2]),
                'bic': data_predict[3]
            })
        
        output_path = os.path.join(self.path_base, f'{index}_metrics.xlsx')

        df_metrics = pd.DataFrame(list_metrics_predict)
        df_metrics.to_excel(output_path, index=False)
        print(df_metrics)


    def diagram_correlation(self, list_predict, test_set, index):
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f'Correlação dos Resíduos das Séries Temporais Autoregressive de {index.title()}', fontsize=16, fontweight='bold')

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
            ax.annotate(f'DW: {dw_stat:.2f}', xy=(0.60, 0.95), xycoords='axes fraction', fontsize=10, 
                        ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='white', facecolor='white'))

            lags = min(10, len(residuos) // 5)  
            acf_values = acf(residuos, nlags=lags, fft=False)  
            q_stat_values, p_values = q_stat(acf_values[1:], len(residuos))  

            ax.annotate(f'Ljung-Box p-valor: {p_values[-1]:.4f}', xy=(0.60, 0.85), xycoords='axes fraction', 
                        fontsize=10, ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='white', facecolor='white'))

            plot_acf(residuos, ax=ax, lags=lags, alpha=0.05, title=f'{ordem} - Pearson: {correlacao:.2f}')

            ax.set_title(f'{ordem} - Correlação Pearson: {correlacao:.2f}', fontweight='bold') 
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=10)

            ax.grid(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = os.path.join(self.path_base, f"correlacao_{index}_autoregressive.png")
        plt.savefig(output_path)


    def test_data_times_series(self, list_predict, index):
        list_residuals = []
        list_test_norm_data = []
        list_test_homoge = []

        # Coletando os resíduos de cada modelo predito
        for predict in list_predict:
            residuals = predict[4]
            list_residuals.append(residuals)

        for i, residuals in enumerate(list_residuals):
            stat, p_valor = shapiro(residuals)

      
            list_test_norm_data.append({
                'stat': stat,
                'p_valor ': p_valor,
                'resultado': 'Normal' if p_valor > 0.05 else 'Não é normal'
            })
        
        print(list_test_norm_data)

        # Teste de Levene para homogeneidade de variâncias
        stat, p_value = levene(*list_residuals)
        print("Estatística do teste de Levene:", stat)
        print("p-valor do teste de Levene:", p_value)

        
        list_test_homoge.append({
            'stat': stat,
            'p_value': p_value
        })

        output_path = os.path.join(self.path_base, f'test_data_times_series_{index}.xlsx')

        df_norm_data = pd.DataFrame(list_test_norm_data)
        df_homoge = pd.DataFrame(list_test_homoge)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_norm_data.to_excel(writer, sheet_name='Normalidade', index=False)
            df_homoge.to_excel(writer, sheet_name='Homogeneidade', index=False)
        

        return list_test_norm_data, list_test_homoge
    
    
    def plot_curve_norm(self, list_predict, list_test_norm_data, index):
        fig, axs = plt.subplots(2, 3, figsize=(10, 6))
        fig.suptitle(f'Train Test da {index.title()} Mensal - Autoregressive', fontsize=13, fontweight='bold')

        axs_flat = axs.flat

        for ax, predict, norm_data in zip(axs_flat, list_predict, list_test_norm_data):
            ordem, previsao, aic, bic, residuos = predict

            sns.histplot(residuos, bins=20, color='#333333', kde=False, ax=ax, stat="density", label="Resíduos")

            mu, sigma = np.mean(residuos), np.std(residuos)
            x = np.linspace(min(residuos), max(residuos), 100)
            y = norm.pdf(x, loc=mu, scale=sigma)
            ax.plot(x, y, color='red', label="Curva Normal")

            p_valor = norm_data['p_valor ']
            resultado = norm_data['resultado']

            # ax.annotate(f'Shapiro Wilk: \np-valor={p_valor:.4f}', 
            #         xy=(0.85, 0.20), xycoords='axes fraction', fontsize=8, 
            #         ha='left', va='top', 
            #         bbox=dict(boxstyle='round,pad=0.3', edgecolor='white', facecolor='white'))

            ax.set_title(f'{ordem} - Shapiro Wilk: p-valor={p_valor:.3f}', fontweight='bold', fontsize=8) 
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=2, fontsize=6)

            ax.set_xlabel('Precipitação', fontweight='bold', fontsize=6)
            ax.set_ylabel('Densidade', fontweight='bold', fontsize=6)

            ax.grid(False)


        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = os.path.join(self.path_base, f"test_norm_{index}_autoregressive.png")
        plt.savefig(output_path)

    def test_kruskal_wallis_times_series(self, list_predict, index):
        list_residuals = []
        kruskal_results = {}

        for predict in list_predict:
            residuals = predict[1].dropna()
            list_residuals.append(residuals)

        kruskal_stat, kruskal_p_value = kruskal(*list_residuals)

        kruskal_results['kruskal_stat'] = kruskal_stat
        kruskal_results['p_value'] = kruskal_p_value

        print(kruskal_results)

        if kruskal_p_value < 0.05:
            residuals_flat = [res for group in list_residuals for res in group]

            group_labels = [
                f'Modelo {i + 1}' for i, group in enumerate(list_residuals) for _ in group
            ]

            dunn_results = posthoc_dunn(
                [group_labels, residuals_flat],
                p_adjust="bonferroni"
            )

            output_path = os.path.join(self.path_base, f'test_kruskal_wallis_times_series_{index}.xlsx')

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                pd.DataFrame([kruskal_results]).to_excel(writer, sheet_name='Kruskal-Wallis', index=False)
                dunn_results.to_excel(writer, sheet_name='Dunn Test', index=True)
        
        if kruskal_p_value > 0.05:
            output_path = os.path.join(self.path_base, f'test_kruskal_wallis_times_series_{index}.xlsx')

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                pd.DataFrame([kruskal_results]).to_excel(writer, sheet_name='Kruskal-Wallis', index=False)


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

        #print(train_set, '\n\n')
        #print(test_set)

        for lags in range(1, 12, 2):
            #print(lags)

            ar_model = AutoReg(train_set.to_numpy(), lags=lags, trend='c').fit()
            params = ar_model.params
            predict = params[0]

            for i, coef in enumerate(ar_model.params[1:]):
                predict += coef * test_set.shift(i + 1)
            
            residuos = test_set - predict
            
            list_predict.append((
                f'Ordem {lags}',
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
    tsa = TestTrainAutoregressive()
    tsa.control_time_series()

if __name__ == '__main__':
    main()
