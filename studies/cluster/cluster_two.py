from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import kmeans, vq
from scipy.cluster.hierarchy import dendrogram

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten

plt.style.use('ggplot')


class ClusterTwo:
    
    def __init__(self, df):
        
        self.df = df



    ## Data preprocessing   
        
    def normalization(self, x, y):
        '''
        Retorna o objeto da classe (um dataframe) com duas colunas
        a mais, normalizadas com desvio padrão 1.

            Parameters:
                    x (str): Nome da coluna 
                    y (str): Nome da outra coluna

            Returns:
                    DataFrame (pd.dataframe): Pandas Dataframe com duas colunas normalizadas.
        '''
        
        self.df[f'scaled_{x}'] = whiten(self.df[x])
        
        self.df[f'scaled_{y}'] = whiten(self.df[y])
        
        return self.df
    

    ## Clusterization methods
    
    def hierarchical_clustering(self, x, y, method, metric = 'euclidean', n_cluster = 3, criterion = 'maxclust'):
        '''
        Retorna o objeto da classe (um dataframe) com a coluna dos labels
        adicionada, de acordo com o algorítimo de clusterização Hirarchical.

            Parameters:
                    method (str): Método a ser utilizado (single, complete, average, centroid, median, or ward )

                    metric (str): Metrica a ser utlilizada (default = 'euclidean')

                    n_cluster (int): Número de clusters a ser utilizado (default = 3)

                    criterion (str): Critério a ser utilizado na construção dos clusters labels (default = 'maxclust')

            Returns:
                    DataFrame (pd.dataframe): Pandas Dataframe uma coluna a mais dos clusters labels.
        '''
        
        Z = linkage(self.df[[x, y]], method)

        self.df[f'cluster_hierarchical_labels'] = fcluster(Z, n_cluster, criterion=criterion)
        
        self.Z = Z

        return self.df


    def kmeans_clustering(self, x, y, n_clusters, check_finite = True, ):

        '''
        Retorna o objeto da classe (um dataframe) com a coluna dos labels
        adicionada, de acordo com o algorítimo de clusterização K-Means.

            Parameters:
                    x (str): Nome da coluna 
                    y (str): Nome da outra coluna
                    n_clusters (int): Número de clusters a ser utilizado
                    check_finite (bool): Indica se uma verificação funciona se tiver um NaN no conjunto dos dados (default = True)

            Returns:
                    DataFrame (pd.dataframe): Pandas Dataframe uma coluna a mais dos clusters labels.
        '''
        
        cluster_centers, _ = kmeans(self.df[[x, y]], n_clusters, check_finite = check_finite)
    
        self.df['cluster_kmeans_labels'], _ = vq(self.df[[x, y]], cluster_centers)

        return self.df


    ## Plots 

    def plot_dendrogram(self):
        
        '''
        Apenas plota o dendrograma de acordo com um linkage definido na função hierarchical_clustering. 
        '''
        
        dn = dendrogram(self.Z)

        plt.show()


    def elbow_plot(self, x, y, cluster_range):

        '''
        Apenas plota uma gráfico de linha com abscissa sendo o range de clusters definidos e com a ordenada as distorções totais. 


            Parameters:
                    x (str): Nome da coluna 
                    y (str): Nome da outra coluna
                    cluster_range (range): Um objeto range 

        '''

        distortions = []

        num_clusters = cluster_range

        for i in num_clusters:

            centroids, distortion = kmeans(self.df[[x, y]], i)

            distortions.append(distortion)



        sns.lineplot(x = num_clusters, y = distortions)


    def plot_cluster(self, x, y, hue):
        
        '''
            Apenas plota um scatter plot. 


            Parameters:
                    x (str): Nome da coluna 
                    y (str): Nome da outra coluna
                    hue (str): Nome da coluna que definirá a tonalidade de cada ponto 
            
        '''
       
        sns.scatterplot(x = x, 
                        y = y,
                        hue = hue, 
                        data = self.df,
                        palette = 'tab10')

                        