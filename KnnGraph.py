from typing import Callable
import math, random
from dataclasses import dataclass

@dataclass
class KnnVertex:
   number:int
   x:float
   y:float

class KnnGraph:

   NO_EDGE = -1 #const para representar a não existencia de uma aresta

   n:int  # numero de vertices
   k:int  # cada vértice conectados ao k vizinhos mais próximos
   
   __func_distancia: Callable[[float,float,float,float],float] #função que recebe 4 floats (2 pontos com x,y) e retorna uma distancia em float
   __adj_matrix:list[list[float]] #lista de adjancecia com o peso dos vérticies sendo a distancia
   __vertex_list: list[KnnVertex] #lista dos vértices e suas posições

   def __init__(self,n:int,k:int, func_distancia:Callable[[float,float,float,float],float] | None = None) -> None:
      if (n <= 0):
         raise Exception("n não pode ser igual ou menor que 0")
      if (k <= 0):
         raise Exception("k não pode ser igual ou menor que 0")
      if (k > n):
         raise Exception("k não pode ser maior que n")
      self.n = n
      self.k = k

      if func_distancia is None: #usa distância euclidiana por padrão
         self.__func_distancia = self.__euclidian_dist
      else:
         self.__func_distancia = func_distancia #usa a func de distancia que o usuário colocou
      
      self.__vertex_list = self.__generate_vertexes() #gera a lista de vertices de forma randomica
      self.__adj_matrix = self.__find_k_nearest()
      
   def __euclidian_dist(self,x1:float,y1:float,x2:float,y2:float)->float:
      return math.sqrt( (x1-x2)**2 + (y1-y2)**2)

   def __generate_vertexes(self)->list[KnnVertex]:
      vertexes:list[KnnVertex] = [ 
         KnnVertex(i,random.uniform(0,self.n),random.uniform(0,self.n)) 
         for i in range(self.n)
      ]
      return sorted(vertexes,key=lambda v: (v.x,v.y))

   def __find_k_nearest(self)->list[list[float]]:
      distance_matrix:list[list[float]] = [[0 for y in range(self.n)] for x in range(self.n)]
      
      for main_index,main_vertex in enumerate(self.__vertex_list): #loop pelo vertice principal (linhas da matriz de distancia)
         for neigh_index,neighbor in enumerate(self.__vertex_list): #loop pelas colunas da matriz de distância
            if neigh_index == main_index: #diagonal
               continue
            distance_matrix[main_index][neigh_index] = self.__func_distancia( #calcula distância
               main_vertex.x, main_vertex.y, neighbor.x, neighbor.y
            )
      
      adj_matrix:list[list[float]] = [ [self.NO_EDGE for y in range(self.n)] for x in range(self.n)] #inicia matriz de adjacência

      for index,vertex in enumerate(self.__vertex_list):
         
         distances_and_indexes:list[tuple[int,float]] = list(enumerate(distance_matrix[index])) #pega lista de distancias e seus indexes
         distances_and_indexes = list(filter(lambda x: x[1] != 0,distances_and_indexes)) #filtra distancia 0 (o própio vértice)
         distances_and_indexes.sort(key = lambda x: x[1]) #ordena pela distancia

         k_nearest:list[tuple[int,float]] = distances_and_indexes[:self.k] #k vizinhos mais próximos

         for neigh_index,distance in k_nearest:
            adj_matrix[index][neigh_index] = distance
      
      return adj_matrix

   def get_edge_list(self)->list[tuple]:
    edge_list = []
    for i in range(len(self.__adj_matrix)):
        for j in range(len(self.__adj_matrix[i])):
            if self.__adj_matrix[i][j] != -1:  # Only include valid edges
                edge_list.append((i, j, self.__adj_matrix[i][j]))
    return edge_list

   def get_adj_matrix(self)->list[list[float]]:
      return self.__adj_matrix

   def plot_graph(self)->None:
      import networkx as nx
      import matplotlib.pyplot as plt

      edge_list = self.get_edge_list()
      graph = nx.DiGraph()
      graph.add_weighted_edges_from(edge_list)
      nx.draw(graph)
      plt.show()

   def vertex_exists(self,vertex_num:int)->bool:
      """
      Numero do Vertice entre 0 e n-1
      """
      return False if vertex_num < 0 or vertex_num >= self.n else True

if __name__ == "__main__":
   graph_ = KnnGraph(10,2)
   #graph_.plot_graph()