from KnnGraph import KnnGraph

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

