import pandas as pd
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

class GraphicsData(object):
    path_base = r"C:\Artigos Cientificos\Analise_Historica_MG\Files\Graficos Dados Obtidos"

    def __init__(self):
        print('Obtendo Gráficos dos Dados...')
        pass

    def control_create_graphics(self):
        data_climate, stats_climate = self.get_data_climate()
        data_cultura = self.get_data_serie_historica_cultura()
        print(data_cultura)

        print(data_climate, '\n\n\n', stats_climate)

        self.make_graphics_stats_climate(stats_climate)
        self.make_graphics_graos(data_cultura)

    def make_graphics_stats_climate(self, df):
        print('Gerando gráficos de estatística do clima ...')

        anos = df['ano'].to_numpy()
        precipitacao_mean = df['precipitacao_mean'].to_numpy()
        temperatura_mean = df['temperatura_mean'].to_numpy()

        # Configuração para os dados
        dados = {
            'Precipitação (mm)': precipitacao_mean,
            'Temperatura (°C)': temperatura_mean,
        }

        cores = ['#333333']

        #plt.style.use('seaborn-whitegrid')
        fig, axs = plt.subplots(1, 2, figsize=(14, 6)) 
        fig.suptitle('Variação Climática em Minas Gerais (2013-2023)', fontsize=16, fontweight='bold')

        for ax, (titulo, valores) in zip(axs, dados.items()):
            ax.bar(anos, valores, color='#333333', edgecolor='black', label=titulo)

            ax.set_title(titulo, fontweight='bold')
            ax.set_xlabel('Ano', fontweight='bold')
            ax.set_ylabel('Média', fontweight='bold')
            ax.set_xticks(anos)
            #ax.legend(loc='upper right')
            ax.grid(False)

            if titulo == 'Temperatura (°C)':
                ax.set_ylim(18, 22.5)  # Define os limites do eixo Y
                ax.set_yticks(np.arange(18, 23, 0.5))

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        dir_path = os.path.join(self.path_base, 'Analise_Temporal')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        output_path = os.path.join(dir_path, 'dados_climaticos_stats_barras.png')

        plt.savefig(output_path)
        plt.clf()

    def plotar_grafico(self, dados, anos, titulo, nome_arquivo, unidades, anos_cafe):
        cores = ['#333333']

        #plt.style.use('seaborn-whitegrid')
        fig, axs = plt.subplots(3, 2, figsize=(14, 12))  # 3 linhas, 2 colunas
        axs = axs.flatten()  # Achata os eixos para facilitar o acesso

        for idx, ((produto, valores)) in enumerate(dados.items()):
            unidade = unidades.get(produto.lower(), 'Produção')  # Busca a unidade correspondente

            if 'café' not in produto.lower():
                axs[idx].bar(anos, valores, color='#333333', edgecolor='black', label=produto)
                axs[idx].set_title(produto, fontweight='bold')
                axs[idx].set_xlabel('Ano', fontweight='bold')
                axs[idx].set_ylabel(f'Produção ({unidade})', fontweight='bold')
                axs[idx].set_xticks(anos)
                axs[idx].grid(False)

            if 'café' in produto.lower():
                axs[idx].bar(anos_cafe, valores, color='#333333', edgecolor='black', label=produto)
                axs[idx].set_title(produto, fontweight='bold')
                axs[idx].set_xlabel('Ano', fontweight='bold')
                axs[idx].set_ylabel(f'Produção ({unidade})', fontweight='bold')
                axs[idx].set_xticks(anos_cafe)
                axs[idx].grid(False)

        # Remove os gráficos extras em caso de menos de 6 produtos
        for idx in range(len(dados), len(axs)):
            fig.delaxes(axs[idx])

        plt.suptitle(titulo, fontweight='bold', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajusta espaçamento para o título

        dir_path = os.path.join(self.path_base, 'Analise_Temporal')
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        output_path = os.path.join(dir_path, nome_arquivo)
        plt.savefig(output_path)
        plt.clf()

    def make_graphics_graos(self, df):
        print('Gerando gráficos das séries históricas das culturas...')
        
        culturas = ['feijão', 'milho', 'soja', 'sorgo', 'cana-de-açúcar', 'café']

        feijao = df[df['cultura'] == 'feijão']
        milho = df[df['cultura'] == 'milho']
        soja = df[df['cultura'] == 'soja']
        sorgo = df[df['cultura'] == 'sorgo']
        cana_de_acucar = df[df['cultura'] == 'cana-de-açúcar']
        cafe = df[df['cultura'] == 'café']
        
        # Garantir alinhamento dos anos
        anos = feijao['ano'].to_numpy()
        anos_cafe = cafe['ano'].to_numpy()

        producao_feijao = feijao['producao'].to_numpy()
        producao_milho = milho['producao'].to_numpy()
        producao_soja = soja['producao'].to_numpy()
        producao_sorgo = sorgo['producao'].to_numpy()
        producao_cana_de_acucar = cana_de_acucar['producao'].to_numpy()
        producao_cafe = cafe['producao'].to_numpy()

        producoes = {
            'Feijão': producao_feijao,
            'Milho': producao_milho,
            'Soja': producao_soja,
            'Sorgo': producao_sorgo,
            'Cana-de-Açúcar': producao_cana_de_acucar,
            'Café': producao_cafe
        }

        # Criar dicionário com unidades
        unidades = {cultura: df[df['cultura'] == cultura]['producao_unidade'].iloc[0] for cultura in culturas}

        # Gerar gráficos
        self.plotar_grafico(producoes, anos, 'Produção das Principais Culturas (2013-2023)', 'producao_principais_culturas.png', unidades, anos_cafe)

    def get_data_climate(self):
        sql_data = f'''
        SELECT
            ano,
            precipitacao,
            temperatura, 
            umidade 
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

        columns = ['ano', 'precipitacao', 'temperatura', 'umidade']
        df = pd.DataFrame(data, columns=columns)

        summary_df = df.groupby('ano').agg(
            precipitacao_mean=('precipitacao', 'mean'),
            precipitacao_max=('precipitacao', 'max'),
            precipitacao_min=('precipitacao', 'min'),
            temperatura_mean=('temperatura', 'mean'),
            temperatura_max=('temperatura', 'max'),
            temperatura_min=('temperatura', 'min'),
            umidade_mean=('umidade', 'mean'),
            umidade_max=('umidade', 'max'),
            umidade_min=('umidade', 'min')
        ).reset_index()

        return df, summary_df
    
    def get_data_serie_historica_cultura(self):
        sql_data = '''
        SELECT
            cultura,
            ano,
            producao,
            producao_unidade
        FROM
            dados_mg.series_historica_cultura
        ORDER BY
            cultura, ano ASC
        '''

        conn = self.conn()
        cursor = conn.cursor()
        cursor.execute(sql_data)
        data = cursor.fetchall()

        columns = ['cultura', 'ano', 'producao', 'producao_unidade']
        df = pd.DataFrame(data, columns=columns)

        return df

    
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
    sds = GraphicsData()
    sds.control_create_graphics()

if __name__ == '__main__':
    main()