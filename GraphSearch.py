from KnnGraph import KnnGraph
import heapq
from dataclasses import dataclass
from typing import Callable


class GraphSearch:

   __graph:KnnGraph
   __n:int

   def __init__(self,graph:KnnGraph) -> None:
      self.__graph = graph
      self.__n = graph.n

   def bfs(self,start_vertex:int,end_vertex:int)->list[int]:
      if not self.__graph.vertex_exists(start_vertex) or not self.__graph.vertex_exists(end_vertex):
         raise Exception(f"Vertice inicial ou final não está contido no grafo com vértices de 0 à {self.__graph.n-1}")
      
      adj_matrix:list[list[float]] = self.__graph.get_adj_matrix()
      visited:list[bool] = [False for x in range(self.__graph.n)]
      parent_of:list[int] = [-1 for x in range(self.__graph.n)]
      queue:list[int] = [start_vertex] #fila para os vértices a serem visitados

      while queue:
         cur_vertex: int = queue.pop(0)
         visited[cur_vertex] =  True
   
         for vertex_index,weight in enumerate(adj_matrix[cur_vertex]):
            if weight != self.__graph.NO_EDGE and not visited[vertex_index]: #aresta existe e não visitamos o vértice ainda
               parent_of[vertex_index] = cur_vertex
               visited[vertex_index] = True
               queue.append(vertex_index)

               if vertex_index == end_vertex: #nó final, retorna o caminho
                  final_path = []
                  cur = end_vertex
                  while cur != -1:
                     final_path.append(cur)
                     cur = parent_of[cur]
                  return final_path[::-1] 
                  
      return [] #não achou o caminho
      
   def dfs(self,start_vertex:int,end_vertex:int)->list[int]:
      if not self.__graph.vertex_exists(start_vertex) or not self.__graph.vertex_exists(end_vertex):
         raise Exception(f"Vertice inicial ou final não está contido no grafo com vértices de 0 à {self.__graph.n-1}")
      
      adj_matrix:list[list[float]] = self.__graph.get_adj_matrix()
      visited:list[bool] = [False for x in range(self.__graph.n)]
      parent_of:list[int] = [-1 for x in range(self.__graph.n)]
      stack:list[int] = [start_vertex] #fila para os vértices a serem visitados

      while stack:
         cur_vertex: int = stack.pop()

         if not visited[cur_vertex]:
            visited[cur_vertex]=  True
            if cur_vertex == end_vertex: #nó final, retorna o caminho
               final_path = []
               cur = end_vertex
               while cur != -1:
                  final_path.append(cur)
                  cur = parent_of[cur]
               return final_path[::-1] 
   
            for vertex_index,weight in enumerate(adj_matrix[cur_vertex]):
               if weight != self.__graph.NO_EDGE and not visited[vertex_index]: #aresta existe e não visitamos o vértice ainda
                  parent_of[vertex_index] = cur_vertex
                  stack.append(vertex_index)

      return [] #não achou o caminho
   
   def __weighted_graph_search(self,start_vertex:int,end_vertex:int, heuristic_fn:Callable[[int,int],float]| None = None)->list[int]:
      search_heuristic_fn: Callable[[int, int], float] = heuristic_fn if heuristic_fn is not None else lambda x, y: 0

      INF = float('inf')
      @dataclass
      class VertexInfo:
         parent:int = -1
         weight:float = INF
         explored:bool = False

      vertex_list:list[VertexInfo] = [VertexInfo() for x in range(self.__n)] #
      vertex_list[start_vertex].weight = 0 #atualiza o peso do primeiro cara
      path_found: bool = False
      adj_matrix:list[list[float]] = self.__graph.get_adj_matrix()

      heap:list[tuple[float,int]] = []
      heapq.heappush(heap,(0,start_vertex))

      while heap:
         cur_weight,cur_index, = heapq.heappop(heap) #pop vertice com menor peso
         cur_vertex:VertexInfo = vertex_list[cur_index] #acha o objeto vertex associado

         if cur_vertex.explored: #ja foi explorado
            continue
         
         cur_vertex.explored = True #marca como explorado
         if cur_index == end_vertex: #vertice final
            path_found = True
            break
         #relaxamento
         for vertex_index,weight in enumerate(adj_matrix[cur_index]): #loop pelos vizinhos
            if weight != self.__graph.NO_EDGE and not vertex_list[vertex_index].explored: #aresta existe e não visitamos o vértice ainda
               new_dist = cur_vertex.weight + adj_matrix[cur_index][vertex_index] #distancia do nó pai + aresta
               old_dist = vertex_list[vertex_index].weight #distancia antiga

               if new_dist < old_dist: #nova distancia é menor que a anterior
                  vertex_list[vertex_index].parent = cur_index #atualiza parente
                  vertex_list[vertex_index].weight = new_dist #atualiza distancia

                  heuristic_weight:float = new_dist + search_heuristic_fn(vertex_index,end_vertex)
                  heapq.heappush(heap,(heuristic_weight,vertex_index)) #push na heap de uma tupla com o novo peso e o index do vertice antigo

      if path_found:
         path = []
         cur = end_vertex
         while cur != -1:
            path.append(cur)
            cur = vertex_list[cur].parent
         path.reverse()
         return path
      else:
         return []
         
   def dijkstra(self,start_vertex:int,end_vertex:int)->list[int]:
      return self.__weighted_graph_search(start_vertex,end_vertex)
      
   def a_star(self,start_vertex:int,end_vertex:int)->list[int]:
      return self.__weighted_graph_search(start_vertex,end_vertex,self.__graph.euclidian_dist_between)

   def plot_path(self,path:list[int])->None:
      if not path:
         print("Caminho não existe, não é possível desenhar")
         return
      import networkx as nx
      import matplotlib
      matplotlib.use('Qt5Agg') 
      import matplotlib.pyplot as plt

      graph = nx.DiGraph()  # Use DiGraph for directed graphs
      graph.add_weighted_edges_from(self.__graph.get_edge_list())

      # Step 2: Generate the path edges from the list of vertices
      path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

      pos = nx.spring_layout(graph)  # Layout for positioning the nodes

      nx.draw(graph, pos, with_labels=True, node_color="lightblue", node_size=700, font_size=10)

      nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), width=1.0, alpha=0.5)

      nx.draw_networkx_edges(graph, pos, edgelist=path_edges, width=2.5, edge_color="red", style="solid", arrowstyle="-|>", arrowsize=25)

      nx.draw_networkx_nodes(graph, pos, nodelist=path, node_color="yellow", node_size=800)

      edge_labels = nx.get_edge_attributes(graph, 'weight')
      formatted_edge_labels = {(u, v): f"{d:.2f}" for (u, v), d in edge_labels.items()}  # Format to 2 decimal places
      nx.draw_networkx_edge_labels(graph, pos, edge_labels=formatted_edge_labels)

      plt.show()
      
   def hill_climbing(self,start_vertex:int,end_vertex:int)->list[int]:
      parent_of:list[int] = [-1 for x in range(self.__n)] #lista de parentes para reconstruir caminho
      adj_matrix:list[list[float]] = self.__graph.get_adj_matrix() # pega a lista de adjacência
      cur_vertex:int = start_vertex #começamos no vertice inicial
      visited:list[bool] = [False for x in range(self.__n)]
      
      path_found = False #flag para dizer se a busca teve sucesso
      while True:
         if cur_vertex == end_vertex: #chegamos no final
            path_found =  True
            break

         visited[cur_vertex] = True
         children:list[tuple[int,float]] = list(
            filter(lambda x: x[1] != self.__graph.NO_EDGE and not visited[x[0]], #filtra pelos filhos não visitados e com arestas válidas
                  enumerate(adj_matrix[cur_vertex])
            )                                             
         ) #filhos são uma lista de tupla (index vertice,peso/dist da aresta)
         if not children: #não tem aresta filho, busca acabou
            break
         
         children_dist_to_goal:list[tuple[int,float]] = list( #vai ser uma lista de tuplas (index vertice, dist para objetivo)
            map(lambda x: (x[0], self.__graph.euclidian_dist_between(x[0],end_vertex)),children)
         )
         children_sorted = sorted(children_dist_to_goal,key=lambda x: x[1]) #ordena pela distancia até no final
         best_children:int = children_sorted[0][0] #index do melhor filho   
         parent_of[best_children] = cur_vertex
         cur_vertex = best_children
      
      if path_found:
         cur = end_vertex
         path = []

         while cur != -1:
            path.append(cur)
            cur = parent_of[cur]
         path.reverse()
         return path
      else:
         return []
         
   def best_first(self,start_vertex:int,end_vertex:int)->list[int]:
      search_heuristic_fn: Callable[[int, int], float] = self.__graph.euclidian_dist_between
      INF = float('inf')
      @dataclass
      class VertexInfo:
         parent:int = -1
         weight:float = INF
         explored:bool = False

      vertex_list:list[VertexInfo] = [VertexInfo() for x in range(self.__n)] #
      vertex_list[start_vertex].weight = 0 #atualiza o peso do primeiro cara
      path_found: bool = False
      adj_matrix:list[list[float]] = self.__graph.get_adj_matrix()

      heap:list[tuple[float,int]] = []
      heapq.heappush(heap,(0,start_vertex))
      parent_vertex:int = -1

      while heap:
         cur_weight,cur_index, = heapq.heappop(heap) #pop vertice com menor peso
         cur_vertex:VertexInfo = vertex_list[cur_index] #acha o objeto vertex associado
         cur_vertex.parent = parent_vertex
         parent_vertex = cur_index

         if cur_vertex.explored: #ja foi explorado
            continue
         
         cur_vertex.explored = True #marca como explorado
         if cur_index == end_vertex: #vertice final
            path_found = True
            break
         #relaxamento
         for vertex_index,weight in enumerate(adj_matrix[cur_index]): #loop pelos vizinhos
            if weight != self.__graph.NO_EDGE and not vertex_list[vertex_index].explored: #aresta existe e não visitamos o vértice ainda
               heuristic_weight:float = search_heuristic_fn(vertex_index,end_vertex)
               heapq.heappush(heap,(heuristic_weight,vertex_index)) #push na heap de uma tupla com o novo peso e o index do vertice antigo

      if path_found:
         path = []
         cur = end_vertex
         while cur != -1:
            path.append(cur)
            cur = vertex_list[cur].parent
         path.reverse()
         return path
      else:
         return []


if __name__ == "__main__":
   graph = KnnGraph(10,2)
   search = GraphSearch(graph)

   bfs_walk = search.best_first(0,7)
   search.plot_path(bfs_walk)