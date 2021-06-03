from scipy.cluster.hierarchy import linkage, fcluster
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten

plt.style.use('ggplot')


class ClusterTwo:
    
    def __init__(self, df):
        
        self.df = df
        
        
    def normalization(self, x, y):
        
        self.df[f'scaled_{x}'] = whiten(self.df[x])
        
        self.df[f'scaled_{y}'] = whiten(self.df[y])
        
        return self.df
    
    
    def hierarchical_clustering(self, method, metric, n_cluster, criterion):
        
        
        Z = linkage(self.df, method)

        self.df[f'cluster_labels_hierarchical'] = fcluster(Z, n_cluster, criterion=criterion)
        
        return self.df
    
    
    def plot_cluster(self, x, y, hue):
       
        sns.scatterplot(x = x, 
                        y = y,
                        hue = hue, 
                        data = self.df,
                        palette = 'tab10')