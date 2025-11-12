import numpy as np
import time
import sys
import os
from mpi4py import MPI

# Función de Carga P(t)
def P(t: float, f: float) -> float:
  """
  Calcula la función de carga pulsante P(t).
  """
  return 0.5 * (1 + np.sign(np.sin(2 * np.pi * f * t)))

# Función Principal de Simulación (MPI)
def run_simulation_mpi(comm, N, L, alpha, dt, t_final, f):
  """
  Ejecuta la simulación MPI de transferencia de calor para una frecuencia f dada.
  
  Retorna (solo en rank 0):
  - T_p: Tiempo de ejecución de la simulación [s]
  - time_history: Array de tiempos [s]
  - T_center_history: Array de temperaturas del punto central [K]
  """
  
  # 1. Inicialización MPI y Geometría
  rank = comm.Get_rank()
  size = comm.Get_size()

  h = L / N
  mu = (alpha * dt) / (h**2)
  n_steps = int(t_final / dt)
  
  local_N = N // size
  start_row = rank * local_N
  
  if rank == 0:
    print(f"Iniciando Simulación MPI (f={f} Hz, size={size})")

  if mu > 0.25:
    if rank == 0:
      print("ADVERTENCIA: Simulación INESTABLE (mu > 0.25)")
    comm.Abort() # Detener todos los procesos si es inestable
      
  # 2. Inicialización de Grillas Locales
  T_local = np.zeros((local_N + 2, N), dtype=np.float64) # +2 celdas fantasma
  S_mask_local = np.zeros((local_N + 2, N), dtype=np.float64)

  global_i_min, global_i_max = int(0.4 * N), int(0.6 * N)
  j_min, j_max = int(0.4 * N), int(0.6 * N)

  i_start = max(0, global_i_min - start_row) + 1  # +1 por fantasma
  i_end = min(local_N, global_i_max - start_row) + 1 # +1 por fantasma

  if i_start < i_end:
    S_mask_local[i_start:i_end, j_min:j_max] = 145.0

  # 3. Definir Vecinos
  up_neighbor = rank - 1 if rank > 0 else MPI.PROC_NULL
  down_neighbor = rank + 1 if rank < size - 1 else MPI.PROC_NULL

  # 4. Lógica de Recolección
  # Encontrar qué rank posee el centro
  center_row_global = N // 2
  center_col_local = N // 2
  center_rank = center_row_global // local_N
  local_center_row_idx = (center_row_global % local_N) + 1 # +1 por fantasma
  
  T_center_history = None
  time_history = np.linspace(0, t_final, n_steps) # Todos calculan esto

  if rank == center_rank:
    T_center_history = np.zeros(n_steps, dtype=np.float64) # Solo 1 rank almacena

  # 5. Bucle Principal de Simulación
  comm.Barrier()
  start_time = MPI.Wtime()

  for n in range(n_steps):
    # 1. Comunicación (Intercambio de Halos)
    comm.Sendrecv(sendbuf=T_local[1, :], dest=up_neighbor,
                  recvbuf=T_local[0, :], source=up_neighbor)
    comm.Sendrecv(sendbuf=T_local[-2, :], dest=down_neighbor,
                  recvbuf=T_local[-1, :], source=down_neighbor)
    
    t = n * dt
    Pt = P(t, f)

    # 2. Cálculo (Vectorizado)
    T_laplacian_local = (
      T_local[2:,   1:-1] +
      T_local[0:-2, 1:-1] + 
      T_local[1:-1, 2:  ] +
      T_local[1:-1, 0:-2] - 
      4 * T_local[1:-1, 1:-1]
    )
    S_internal_local = S_mask_local[1:-1, 1:-1] * Pt

    # 3. Actualización
    T_local[1:-1, 1:-1] += mu * T_laplacian_local + S_internal_local * dt

    # 4. Recolección de datos (solo el rank propietario)
    if rank == center_rank:
        T_center_history[n] = T_local[local_center_row_idx, center_col_local]

  comm.Barrier()
  end_time = MPI.Wtime()
  T_p = end_time - start_time

  # 6. Recolección Final de Datos en Rank 0
  if rank == 0:
    if center_rank != 0:
      T_center_history = np.zeros(n_steps, dtype=np.float64)
      comm.Recv(T_center_history, source=center_rank, tag=11)

    print(f"Simulación (f={f} Hz) completada en {T_p:.4f} segundos ({T_p/60:.4f} minutos).")

    output_dir = "./media"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"mpi_results_f_{f:.2f}_p_{size}.npz")

    np.savez(output_file,
             time=time_history,
             temp=T_center_history,
             T_exec=T_p)
    print(f"Resultados guardados en: {output_file}\n")
      
  elif rank == center_rank:
    # El rank propietario envía sus datos al rank 0
    comm.Send(T_center_history, dest=0, tag=11)

# Bloque de Ejecución Principal
if __name__ == "__main__":
  
  # 1. Leer Frecuencia desde Argumentos de Línea de Comandos
  if len(sys.argv) != 2:
    if MPI.COMM_WORLD.Get_rank() == 0:
      print("Error: El script debe llamarse con una frecuencia como argumento.")
      print(f"Ejemplo: mpiexec -n 8 python {sys.argv[0]} 0.5")
    MPI.COMM_WORLD.Abort()
      
  f_arg = float(sys.argv[1]) # Leer f (0.5 o 10.0)

  # 2. Parámetros Globales
  N = 512
  L = 0.05
  alpha = 1.1e-4
  dt = 2.0e-5
  t_final = 5.0

  # 3. Inicializar MPI y Ejecutar Simulación
  comm = MPI.COMM_WORLD
  run_simulation_mpi(comm, N, L, alpha, dt, t_final, f=f_arg)