import random
from typing import Tuple

def randomWalk_Laplace(args: Tuple[int, int, int, int]) -> float:
  """
  Realiza una caminata aleatoria para estimar el potencial en (i, j).
  
  Args (tupla para compatibilidad con mp.map):
    i_start, j_start: Coordenadas iniciales
    N: Tamaño de la grilla
    M: Número de caminantes
  """
  i_start, j_start, N, M = args

  # --- 1. Condiciones de Borde (Valores fijos) ---
  # Pared Superior (V=100)
  if i_start == 0:
    return 100.0
  # Pared Inferior (V=0)
  if i_start == N - 1:
    return 0.0
  # Pared Izquierda (V=0)
  if j_start == 0:
    return 0.0
  # Pared Derecha (V=0)
  if j_start == N - 1:
    return 0.0

  # --- 2. Caminata Aleatoria (Monte Carlo) ---
  potencial_acumulado = 0.0
  
  # Movimientos posibles: (di, dj) -> Arriba, Abajo, Izquierda, Derecha
  moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

  for _ in range(M):
    curr_i, curr_j = i_start, j_start
    
    while True:
      # Seleccionar movimiento al azar
      move = random.choice(moves)
      curr_i += move[0]
      curr_j += move[1]

      # Verificar si chocó con algún borde
      # Borde Superior
      if curr_i == 0:
        potencial_acumulado += 100.0
        break
      # Otros Bordes (V=0) -> Termina la caminata, suma 0
      if curr_i == N - 1 or curr_j == 0 or curr_j == N - 1:
        potencial_acumulado += 0.0
        break
  
  # Retornar el promedio
  return potencial_acumulado / M