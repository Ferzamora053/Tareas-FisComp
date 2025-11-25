import multiprocessing as mp
import numpy as np
import os
import time
import random
from typing import Tuple
from numpy.typing import NDArray
from random_walk_laplace import randomWalk_Laplace

def _init_worker(seed_offset: int = 0):
  """Inicializador para cada worker: siembra RNG por proceso."""
  random.seed(seed_offset + os.getpid() + int(time.time() % 1_000_000))

def run_parallel_simulation(N: int, M: int, num_procs: int) -> Tuple[NDArray[np.float64], float]:
  """
  Configura y ejecuta la simulación paralela.
  Si num_procs = 1, ejecuta en serie (sin Pool) para un T1 razonable.
  Retorna: (Matriz de Potencial, Tiempo de ejecución)
  """
  
  # Generamos la lista de tareas (coordenadas a resolver)
  # Solo necesitamos resolver los puntos interiores (1..N-2)
  tasks = []
  for i in range(1, N - 1):
    for j in range(1, N - 1):
      tasks.append((i, j, N, M))

  start_time = time.time()

  # Crear el pool de procesos y ejecutar las tareas
  # 'chunksize' ayuda a reducir el overhead de comunicación
  if num_procs == 1:
    # Ruta serial: evita la sobrecarga de Pool para medición T1 real
    results = [randomWalk_Laplace(t) for t in tasks]
  else:
    # Ruta paralela con Pool y initializer para sembrar RNG
    with mp.Pool(processes=num_procs, initializer=_init_worker, initargs=(12345,)) as pool:
        results = pool.map(randomWalk_Laplace, tasks)

   # Reconstruir la matriz de potencial 2D a partir de la lista plana de resultados
  V_parallel = np.zeros((N, N), dtype=np.float64)
  V_parallel[0, :] = 100.0  # Condición de borde superior
  # Los otros bordes ya son 0.0 por defecto en np.zeros
  
  # Los resultados están en el mismo orden que las tareas
  idx = 0
  for i in range(1, N - 1):
    for j in range(1, N - 1):
      V_parallel[i, j] = results[idx]
      idx += 1

  end_time = time.time()
  total_time = end_time - start_time
  
  return V_parallel, total_time

# Bloque para probar este archivo por sí solo si fuera necesario
if __name__ == "__main__":
  # Prueba rápida
  N_test = 20
  M_test = 100
  cores = 4
  v, t = run_parallel_simulation(N_test, M_test, cores)
  print("Prueba exitosa.")