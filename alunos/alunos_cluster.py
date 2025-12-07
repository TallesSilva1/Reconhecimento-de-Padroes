import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*80)
print("ANÁLISE MULTIVARIADA DE DADOS EDUCACIONAIS - CENSO 2018")
print("="*80)

# ============================================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# ============================================================================

def carregar_dados():
    """Carrega dados do censo educacional completos da pasta dados_alunos"""
    print("Tentando carregar arquivo completo DM_ALUNO.CSV...")
    
    try:
        # Carregar arquivo completo - todos os dados mesmo
        df = pd.read_csv('dados_alunos/DM_ALUNO.CSV', sep='|', encoding='latin1', low_memory=False)
        print("DADOS COMPLETOS CARREGADOS com sucesso!")
        print(f"Total de registros: {df.shape[0]:,}")
        print(f"Total de colunas: {df.shape[1]}")
        print(f"Tamanho aproximado: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        return df
        
    except FileNotFoundError:
        print("ERRO: Arquivo DM_ALUNO.CSV não encontrado na pasta dados_alunos/")
        print("Verificando arquivos disponíveis...")
        
        import os
        try:
            files = os.listdir('dados_alunos/')
            print("Arquivos encontrados:")
            for file in files:
                print(f"  - {file}")
        except:
            print("Pasta dados_alunos/ não encontrada")
        
        raise FileNotFoundError("Não foi possível carregar os dados. Verifique se o arquivo DM_ALUNO.CSV existe em dados_alunos/")
    
    except Exception as e:
        print(f"ERRO ao carregar dados: {str(e)}")
        print("Dica: Verifique se o arquivo não está corrompido ou sendo usado por outro programa")
        raise

def preprocessar_dados(df):
    """Limpa e prepara todos os dados para análise"""
    print("\n" + "="*60)
    print("PRÉ-PROCESSAMENTO DOS DADOS COMPLETOS")
    print("="*60)
    
    print(f"Dataset original: {len(df):,} registros")
    
    # 1. Limpeza inicial - tirando o lixo
    print("Limpando dados...")
    
    # Remover registros com dados essenciais faltando
    essenciais = ['NU_IDADE', 'TP_SEXO', 'TP_SITUACAO']
    df_inicial = len(df)
    for col in essenciais:
        if col in df.columns:
            df = df[df[col].notna()]
    
    print(f"Registros após limpeza: {len(df):,} (removidos: {df_inicial - len(df):,})")
    
    # 2. Criar variáveis categóricas com nomes que fazem sentido
    print("Criando variáveis categóricas...")
    
    mappings = {
        'TP_SEXO': {1: 'Feminino', 2: 'Masculino'},
        'TP_COR_RACA': {1: 'Branca', 2: 'Preta', 3: 'Parda', 4: 'Amarela', 5: 'Indígena', 0: 'Não informado'},
        'TP_SITUACAO': {2: 'Cursando', 3: 'Trancado', 4: 'Desvinculado', 5: 'Transferido', 6: 'Formado', 7: 'Falecido'},
        'TP_TURNO': {1: 'Matutino', 2: 'Vespertino', 3: 'Noturno', 4: 'Integral'},
        'TP_CATEGORIA_ADMINISTRATIVA': {1: 'Pública Federal', 2: 'Pública Estadual', 3: 'Pública Municipal', 
                                      4: 'Privada com fins lucrativos', 5: 'Privada sem fins lucrativos', 
                                      7: 'Especial', 8: 'Privada - Particular em sentido estrito', 
                                      9: 'Privada - Comunitária'}
    }
    
    for coluna, mapping in mappings.items():
        if coluna in df.columns:
            df[coluna + '_DESC'] = df[coluna].map(mapping).fillna('Não informado')
    
    # 3. Calcular algumas variáveis derivadas úteis
    print("Calculando variáveis derivadas...")
    
    # Idade de ingresso (aproximada)
    if 'NU_ANO_INGRESSO' in df.columns and 'NU_IDADE' in df.columns:
        df['IDADE_INGRESSO'] = df['NU_IDADE'] - (2018 - df['NU_ANO_INGRESSO'])
        df['IDADE_INGRESSO'] = df['IDADE_INGRESSO'].clip(lower=15, upper=70)  # Limites razoáveis
    
    # Faixas etárias pra análise
    if 'NU_IDADE' in df.columns:
        df['FAIXA_ETARIA'] = pd.cut(df['NU_IDADE'], 
                                   bins=[0, 20, 25, 30, 40, 100], 
                                   labels=['<=20', '21-25', '26-30', '31-40', '>40'])
    
    # Tempo no curso
    if 'NU_ANO_INGRESSO' in df.columns:
        df['TEMPO_CURSO'] = 2018 - df['NU_ANO_INGRESSO']
        df['TEMPO_CURSO'] = df['TEMPO_CURSO'].clip(lower=0, upper=15)  # Máximo 15 anos faz sentido
    
    # Indicador de conclusão
    if 'IN_CONCLUINTE' in df.columns:
        df['CONCLUIU'] = df['IN_CONCLUINTE'].fillna(0)
    elif 'TP_SITUACAO' in df.columns:
        df['CONCLUIU'] = (df['TP_SITUACAO'] == 6).astype(int)
    
    # 4. Filtros de qualidade - tirando outliers malucos
    print("Aplicando filtros de qualidade...")
    
    # Idades razoáveis
    if 'NU_IDADE' in df.columns:
        df = df[(df['NU_IDADE'] >= 15) & (df['NU_IDADE'] <= 100)]
    
    # Anos de ingresso válidos
    if 'NU_ANO_INGRESSO' in df.columns:
        df = df[(df['NU_ANO_INGRESSO'] >= 1990) & (df['NU_ANO_INGRESSO'] <= 2018)]
    
    # 5. Otimização de memória para datasets grandes
    if len(df) > 100000:
        print(f"Dataset grande detectado ({len(df):,} registros)")
        print("Otimizando tipos de dados pra economizar memória...")
        
        # Converter para tipos mais eficientes
        for col in df.columns:
            if df[col].dtype == 'int64':
                if df[col].min() >= 0 and df[col].max() <= 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].min() >= -32768 and df[col].max() <= 32767:
                    df[col] = df[col].astype('int16')
                elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                    df[col] = df[col].astype('int32')
    
    # 6. Resumo final
    categoricas_criadas = len([c for c in df.columns if c.endswith('_DESC')])
    print(f"\nPRÉ-PROCESSAMENTO CONCLUÍDO:")
    print(f"   Registros finais: {len(df):,}")
    print(f"   Colunas totais: {df.shape[1]}")
    print(f"   Variáveis categóricas criadas: {categoricas_criadas}")
    print(f"   Uso de memória: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Estatísticas básicas
    if 'NU_IDADE' in df.columns:
        print(f"   Idade: {df['NU_IDADE'].min()}-{df['NU_IDADE'].max()} anos (média: {df['NU_IDADE'].mean():.1f})")
    
    if 'TP_SEXO_DESC' in df.columns:
        dist_sexo = df['TP_SEXO_DESC'].value_counts()
        print(f"   Sexo: {dist_sexo.to_dict()}")
    
    return df

# ============================================================================
# 2. ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# ============================================================================

def analise_exploratoria(df):
    """Realiza análise exploratória completa"""
    print("\n" + "="*60)
    print("ANÁLISE EXPLORATÓRIA DE DADOS")
    print("="*60)
    
    # Estatísticas descritivas
    print("\nESTATÍSTICAS DESCRITIVAS:")
    if 'NU_IDADE' in df.columns:
        print(f"Idade: {df['NU_IDADE'].mean():.1f} ± {df['NU_IDADE'].std():.1f} anos")
        print(f"Faixa: {df['NU_IDADE'].min()}-{df['NU_IDADE'].max()} anos")
    
    if 'TEMPO_CURSO' in df.columns:
        print(f"Tempo no curso: {df['TEMPO_CURSO'].mean():.1f} ± {df['TEMPO_CURSO'].std():.1f} anos")
    
    # Distribuições principais
    print("\nDISTRIBUIÇÕES:")
    for col in ['TP_SEXO_DESC', 'TP_SITUACAO_DESC', 'FAIXA_ETARIA']:
        if col in df.columns:
            dist = df[col].value_counts()
            print(f"\n{col}:")
            for categoria, count in dist.head(5).items():
                pct = (count/len(df))*100
                print(f"  {categoria}: {count} ({pct:.1f}%)")
    
    # Gráficos principais
    criar_graficos_eda(df)
    
    return df

def criar_graficos_eda(df):
    """Cria visualizações para EDA"""
    
    # Configurar subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Distribuição de Idade', 'Situação no Curso', 
                       'Sexo por Faixa Etária', 'Evolução Temporal'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Histograma de idade
    if 'NU_IDADE' in df.columns:
        fig.add_trace(
            go.Histogram(x=df['NU_IDADE'], name='Idade', showlegend=False),
            row=1, col=1
        )
    
    # 2. Situação no curso
    if 'TP_SITUACAO_DESC' in df.columns:
        situacao_counts = df['TP_SITUACAO_DESC'].value_counts()
        fig.add_trace(
            go.Bar(x=situacao_counts.index, y=situacao_counts.values, 
                  name='Situação', showlegend=False),
            row=1, col=2
        )
    
    # 3. Sexo por faixa etária
    if 'TP_SEXO_DESC' in df.columns and 'FAIXA_ETARIA' in df.columns:
        crosstab = pd.crosstab(df['FAIXA_ETARIA'], df['TP_SEXO_DESC'])
        for sexo in crosstab.columns:
            fig.add_trace(
                go.Bar(x=crosstab.index, y=crosstab[sexo], name=sexo),
                row=2, col=1
            )
    
    # 4. Evolução temporal
    if 'NU_ANO_INGRESSO' in df.columns:
        ingressos = df['NU_ANO_INGRESSO'].value_counts().sort_index()
        fig.add_trace(
            go.Scatter(x=ingressos.index, y=ingressos.values, 
                      mode='lines+markers', name='Ingressos', showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="Análise Exploratória - Dados Educacionais")
    fig.write_html("eda_completa.html")
    fig.show()

# ============================================================================
# 3. ANÁLISE DE CORRELAÇÕES E ASSOCIAÇÕES
# ============================================================================

def analise_correlacoes(df):
    """Analisa correlações entre variáveis"""
    print("\n" + "="*60)
    print("ANÁLISE DE CORRELAÇÕES")
    print("="*60)
    
    # Selecionar variáveis numéricas
    numericas = df.select_dtypes(include=[np.number]).columns
    numericas = [col for col in numericas if df[col].nunique() > 1]
    
    if len(numericas) < 2:
        print("Poucas variáveis numéricas para correlação")
        return df
    
    # Matriz de correlação
    corr_matrix = df[numericas].corr()
    
    # Visualização
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Matriz de Correlação - Variáveis Educacionais')
    plt.tight_layout()
    plt.savefig('matriz_correlacao.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Top correlações
    print("\nCORRELAÇÕES MAIS FORTES:")
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlacoes = upper_triangle.unstack().dropna()
    correlacoes_abs = correlacoes.abs().sort_values(ascending=False)
    
    for (var1, var2), corr in correlacoes_abs.head(10).items():
        print(f"  {var1} ↔ {var2}: {corr:.3f}")
    
    return df

# ============================================================================
# 4. MODELAGEM PREDITIVA - CLASSIFICAÇÃO
# ============================================================================

def modelagem_preditiva(df):
    """Implementa modelos de classificação"""
    print("\n" + "="*60)
    print("MODELAGEM PREDITIVA - CLASSIFICAÇÃO")
    print("="*60)
    
    # Preparar dados para modelagem
    if 'CONCLUIU' not in df.columns:
        print("Variável target 'CONCLUIU' não encontrada")
        return df
    
    # Features para o modelo
    features_numericas = ['NU_IDADE', 'TEMPO_CURSO']
    features_numericas = [f for f in features_numericas if f in df.columns]
    
    # Encoding de variáveis categóricas
    le = LabelEncoder()
    features_categoricas = []
    
    for col in ['TP_SEXO', 'TP_TURNO', 'TP_CATEGORIA_ADMINISTRATIVA']:
        if col in df.columns:
            df[f'{col}_ENCODED'] = le.fit_transform(df[col].fillna(-1))
            features_categoricas.append(f'{col}_ENCODED')
    
    # Dataset final
    features = features_numericas + features_categoricas
    features = [f for f in features if f in df.columns]
    
    if len(features) < 2:
        print("Poucas features disponíveis para modelagem")
        return df
    
    # Preparar X e y
    X = df[features].fillna(df[features].mean())
    y = df['CONCLUIU']
    
    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Modelos
    modelos = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    print(f"\nAVALIAÇÃO DOS MODELOS (Target: Conclusão do Curso)")
    print(f"Features utilizadas: {features}")
    print(f"Dados: {X_train.shape[0]} treino, {X_test.shape[0]} teste")
    
    resultados = {}
    
    for nome, modelo in modelos.items():
        # Treinar
        modelo.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=5)
        
        # Predição
        y_pred = modelo.predict(X_test)
        
        # Métricas
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        resultados[nome] = {
            'CV Score': cv_scores.mean(),
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        
        print(f"\n{nome}:")
        print(f"  CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1-Score: {f1:.3f}")
    
    # Feature importance (Random Forest)
    if 'Random Forest' in modelos:
        rf_model = modelos['Random Forest']
        importances = rf_model.feature_importances_
        
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)[::-1]
        plt.bar(range(len(features)), importances[indices])
        plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
        plt.title('Importância das Features - Random Forest')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return df, resultados

# ============================================================================
# 5. CLUSTERING E SEGMENTAÇÃO
# ============================================================================

def analise_clustering(df):
    """Realiza clustering de estudantes"""
    print("\n" + "="*60)
    print("CLUSTERING E SEGMENTAÇÃO")
    print("="*60)
    
    # Preparar dados para clustering
    features_clustering = ['NU_IDADE', 'TEMPO_CURSO']
    features_clustering = [f for f in features_clustering if f in df.columns]
    
    if len(features_clustering) < 2:
        print("Poucas variáveis para clustering")
        return df
    
    # Dataset para clustering
    X_cluster = df[features_clustering].fillna(df[features_clustering].mean())
    
    # Padronizar dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Encontrar número ótimo de clusters (Elbow Method)
    inertias = []
    silhouette_scores = []
    K_range = range(2, min(8, len(X_scaled)//2))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        if len(set(labels)) > 1:
            silhouette_scores.append(silhouette_score(X_scaled, labels))
        else:
            silhouette_scores.append(0)
    
    # Melhor k por silhouette score
    best_k = K_range[np.argmax(silhouette_scores)]
    
    print(f"\nCLUSTERING K-MEANS:")
    print(f"Número ótimo de clusters: {best_k}")
    print(f"Silhouette Score: {max(silhouette_scores):.3f}")
    
    # Clustering final
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(X_scaled)
    df['CLUSTER'] = clusters
    
    # Análise dos clusters
    print(f"\nPERFIL DOS CLUSTERS:")
    for cluster in range(best_k):
        mask = df['CLUSTER'] == cluster
        cluster_data = df[mask]
        
        print(f"\nCluster {cluster} (n={len(cluster_data)}):")
        
        if 'NU_IDADE' in df.columns:
            print(f"  Idade média: {cluster_data['NU_IDADE'].mean():.1f} anos")
        
        if 'TEMPO_CURSO' in df.columns:
            print(f"  Tempo no curso: {cluster_data['TEMPO_CURSO'].mean():.1f} anos")
        
        if 'CONCLUIU' in df.columns:
            taxa_conclusao = cluster_data['CONCLUIU'].mean() * 100
            print(f"  Taxa de conclusão: {taxa_conclusao:.1f}%")
    
    # Visualização dos clusters
    if len(features_clustering) >= 2:
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Clusters no espaço original
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(df[features_clustering[0]], df[features_clustering[1]], 
                            c=df['CLUSTER'], cmap='viridis', alpha=0.7)
        plt.xlabel(features_clustering[0])
        plt.ylabel(features_clustering[1])
        plt.title('Clusters - Espaço Original')
        plt.colorbar(scatter)
        
        # Plot 2: PCA
        if X_scaled.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            plt.subplot(1, 2, 2)
            scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                                 c=df['CLUSTER'], cmap='viridis', alpha=0.7)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            plt.title('Clusters - Espaço PCA')
            plt.colorbar(scatter2)
        
        plt.tight_layout()
        plt.savefig('clusters_analise.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return df

# ============================================================================
# 6. ANÁLISE TEMPORAL E TENDÊNCIAS
# ============================================================================

def analise_temporal(df):
    """Analisa tendências temporais"""
    print("\n" + "="*60)
    print("ANÁLISE TEMPORAL E TENDÊNCIAS")
    print("="*60)
    
    if 'NU_ANO_INGRESSO' not in df.columns:
        print("Dados temporais não disponíveis")
        return df
    
    # Análise por ano de ingresso
    temporal_data = df.groupby('NU_ANO_INGRESSO').agg({
        'NU_IDADE': 'mean',
        'CONCLUIU': 'mean' if 'CONCLUIU' in df.columns else 'count',
        'ID_ALUNO': 'count'
    }).round(2)
    
    temporal_data.columns = ['Idade_Media', 'Taxa_Conclusao', 'Total_Ingressos']
    
    print("\nTENDÊNCIAS POR ANO:")
    print(temporal_data)
    
    # Visualização temporal
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Ingressos por ano
    axes[0,0].plot(temporal_data.index, temporal_data['Total_Ingressos'], marker='o')
    axes[0,0].set_title('Ingressos por Ano')
    axes[0,0].set_ylabel('Número de Ingressos')
    
    # 2. Idade média por ano
    axes[0,1].plot(temporal_data.index, temporal_data['Idade_Media'], marker='s', color='orange')
    axes[0,1].set_title('Idade Média dos Ingressantes')
    axes[0,1].set_ylabel('Idade (anos)')
    
    # 3. Taxa de conclusão por ano
    if 'CONCLUIU' in df.columns:
        axes[1,0].plot(temporal_data.index, temporal_data['Taxa_Conclusao']*100, marker='^', color='green')
        axes[1,0].set_title('Taxa de Conclusão por Ano')
        axes[1,0].set_ylabel('Taxa (%)')
    
    # 4. Distribuição por sexo ao longo do tempo
    if 'TP_SEXO_DESC' in df.columns:
        sexo_temporal = pd.crosstab(df['NU_ANO_INGRESSO'], df['TP_SEXO_DESC'], normalize='index')
        sexo_temporal.plot(kind='bar', stacked=True, ax=axes[1,1], alpha=0.8)
        axes[1,1].set_title('Distribuição por Sexo ao Longo do Tempo')
        axes[1,1].set_ylabel('Proporção')
        axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('analise_temporal.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df, temporal_data

# ============================================================================
# 7. RELATÓRIO FINAL E INSIGHTS
# ============================================================================

def gerar_relatorio_final(df, resultados_modelo=None, temporal_data=None):
    """Gera relatório final com todos os insights"""
    print("\n" + "="*80)
    print("RELATÓRIO FINAL - INSIGHTS E CONCLUSÕES")
    print("="*80)
    
    insights = []
    
    # 1. Demografia
    if 'NU_IDADE' in df.columns:
        idade_media = df['NU_IDADE'].mean()
        insights.append(f"PERFIL DEMOGRÁFICO: Idade média de {idade_media:.1f} anos")
    
    if 'TP_SEXO_DESC' in df.columns:
        dist_sexo = df['TP_SEXO_DESC'].value_counts(normalize=True)
        predominante = dist_sexo.index[0]
        pct = dist_sexo.iloc[0] * 100
        insights.append(f"   Predominância {predominante.lower()}: {pct:.1f}%")
    
    # 2. Situação acadêmica
    if 'TP_SITUACAO_DESC' in df.columns:
        situacao_principal = df['TP_SITUACAO_DESC'].value_counts().index[0]
        insights.append(f"SITUAÇÃO ACADÊMICA: Maioria em situação '{situacao_principal}'")
    
    if 'CONCLUIU' in df.columns:
        taxa_conclusao = df['CONCLUIU'].mean() * 100
        insights.append(f"   Taxa geral de conclusão: {taxa_conclusao:.1f}%")
    
    # 3. Clusters identificados
    if 'CLUSTER' in df.columns:
        n_clusters = df['CLUSTER'].nunique()
        insights.append(f"SEGMENTAÇÃO: {n_clusters} perfis distintos identificados")
        
        # Cluster com maior taxa de conclusão
        if 'CONCLUIU' in df.columns:
            cluster_performance = df.groupby('CLUSTER')['CONCLUIU'].mean()
            melhor_cluster = cluster_performance.idxmax()
            melhor_taxa = cluster_performance.max() * 100
            insights.append(f"   Cluster {melhor_cluster} apresenta maior taxa de conclusão: {melhor_taxa:.1f}%")
    
    # 4. Modelagem preditiva
    if resultados_modelo:
        melhor_modelo = max(resultados_modelo.keys(), key=lambda k: resultados_modelo[k]['F1-Score'])
        melhor_f1 = resultados_modelo[melhor_modelo]['F1-Score']
        insights.append(f"PREDIÇÃO: {melhor_modelo} obteve melhor performance (F1: {melhor_f1:.3f})")
    
    # 5. Tendências temporais
    if temporal_data is not None and len(temporal_data) > 1:
        tendencia_ingressos = "crescente" if temporal_data['Total_Ingressos'].iloc[-1] > temporal_data['Total_Ingressos'].iloc[0] else "decrescente"
        insights.append(f"TENDÊNCIA TEMPORAL: Padrão {tendencia_ingressos} de ingressos")
    
    # Imprimir insights
    print("\nPRINCIPAIS INSIGHTS:")
    for i, insight in enumerate(insights, 1):
        print(f"\n{i}. {insight}")
    
    # Recomendações
    print(f"\nRECOMENDAÇÕES ESTRATÉGICAS:")
    print(f"\n1. RETENÇÃO: Focar nos clusters com menor taxa de conclusão")
    print(f"2. CAPTAÇÃO: Considerar tendências demográficas identificadas")
    print(f"3. MONITORAMENTO: Usar modelos preditivos para identificação precoce de risco")
    print(f"4. PERSONALIZAÇÃO: Adaptar estratégias por perfil de cluster")
    
    return insights

# ============================================================================
# FUNÇÃO PRINCIPAL - EXECUÇÃO COMPLETA
# ============================================================================

def main():
    """Executa análise completa para o artigo usando todos os dados"""
    print("INICIANDO ANÁLISE COMPLETA COM DADOS REAIS DO CENSO 2018")
    print("="*80)
    
    # 1. Carregar todos os dados do arquivo real
    print("ETAPA 1/8: Carregamento de dados...")
    df = carregar_dados()
    
    # 2. Pré-processar dados completos
    print("\nETAPA 2/8: Pré-processamento...")
    df = preprocessar_dados(df)
    
    # 3. Análise exploratória completa
    print("\nETAPA 3/8: Análise exploratória...")
    df = analise_exploratoria(df)
    
    # 4. Análise de correlações
    print("\nETAPA 4/8: Análise de correlações...")
    df = analise_correlacoes(df)
    
    # 5. Modelagem preditiva com dados reais
    print("\nETAPA 5/8: Modelagem preditiva...")
    try:
        df, resultados_modelo = modelagem_preditiva(df)
    except Exception as e:
        print(f"Erro na modelagem: {e}")
        resultados_modelo = None
    
    # 6. Clustering com dataset completo
    print("\nETAPA 6/8: Clustering e segmentação...")
    try:
        df = analise_clustering(df)
    except Exception as e:
        print(f"Erro no clustering: {e}")
    
    # 7. Análise temporal completa
    print("\nETAPA 7/8: Análise temporal...")
    try:
        df, temporal_data = analise_temporal(df)
    except Exception as e:
        print(f"Erro na análise temporal: {e}")
        temporal_data = None
    
    # 8. Relatório final com estatísticas reais
    print("\nETAPA 8/8: Gerando relatório final...")
    insights = gerar_relatorio_final(df, resultados_modelo, temporal_data)
    
    # Salvar dataset processado para referência
    print(f"\nSalvando dataset processado...")
    df.to_csv('dataset_processado_completo.csv', index=False)
    
    print(f"\n" + "="*80)
    print("ANÁLISE COMPLETA FINALIZADA COM SUCESSO!")
    print("="*80)
    print(f"Dataset analisado: {len(df):,} registros")
    print(f"Arquivos gerados:")
    print(f"  • eda_completa.html (gráficos interativos)")
    print(f"  • matriz_correlacao.png")
    print(f"  • clusters_analise.png") 
    print(f"  • feature_importance.png")
    print(f"  • analise_temporal.png")
    print(f"  • dataset_processado_completo.csv")
    print(f"\nTodos os números estão prontos para seu artigo de 8 páginas!")
    print(f"Use os insights impressos acima para preencher o template!")
    
    return df, insights

if __name__ == "__main__":
    df_final, insights_finais = main()