import numpy as np
import time
from mpi4py import MPI

# Función de Carga P(t)
def P(t:float, f:float) -> float:
  """
  Calcula la función de carga pulsante P(t). 
  """
  return 0.5 * (1 + np.sign(np.sin(2 * np.pi * f * t)))

## Inicialización de MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

## Definición de parámetros
N = 512         # Resolución de la grilla
L = 0.05        # Largo de la placa [m]
alpha = 1.1e-4  # Difusividad térmica del cobre [m^2/s]
dt = 2.0e-5     # Paso del tiempo [s]
t_final = 5     # Tiempo de simulación
f = 0.5         # Frecuencia de la carga [Hz]

h = L / N       # Paso del espacio [m]
mu = (alpha * dt) / (h**2)
n_steps = int(t_final // dt)

## Cálculo de la geometría local para cada proceso
local_N = N // size # Número de filas locales por proceso
start_row = rank * local_N
end_row = start_row + local_N

## Inicialización de la grilla local y condiciones iniciales
# +2 para las celdas fantasmas superior e inferior
T_local = np.zeros((local_N + 2, N), dtype=np.float64)
S_mask_local = np.zeros((local_N + 2, N), dtype=np.float64)

# Definir la máscara S_mask_local basado en la posición global
global_i_min, global_i_max = int(0.4 * N), int(0.6 * N)
j_min, j_max = int(0.4 * N), int(0.6 * N)

# Convertir i_min e i_max global a índices locales
i_start = max(0, global_i_min - start_row) + 1     # +1 por la celda fantasma
i_end = min(local_N, global_i_max - start_row) + 1 # +1 por la celda fantasma

# Aplicar la máscara solo si la franja del proceso se superpone con la fuente
if i_start < i_end:
  S_mask_local[i_start:i_end, j_min:j_max] = 145.0

## Definir vecinos
up_neighbor = rank - 1
if rank == 0:
  up_neighbor = MPI.PROC_NULL

down_neighbor = rank + 1
if rank == size - 1:
  down_neighbor = MPI.PROC_NULL

comm.Barrier()  # Sincronizar antes de iniciar la simulación
start_time = MPI.Wtime() # Iniciar el temporizador

## Bucle principal de la simulación paralela
for n in range(n_steps):
  # Comunicación de bordes con procesos vecinos
  comm.Sendrecv(sendbuf=T_local[1, :], dest=up_neighbor,
                recvbuf=T_local[0, :], source=up_neighbor)
  comm.Sendrecv(sendbuf=T_local[-2, :], dest=down_neighbor,
                recvbuf=T_local[-1, :], source=down_neighbor)
  
  # Calcular la carga actual P(t)
  t = n * dt
  Pt = P(t, f)

  # Calcular el Laplaciano discreto de forma vectorizada
  T_laplacian_local = (
    T_local[2:  , 1:-1] +
    T_local[0:-2, 1:-1] +
    T_local[1:-1, 2:  ] +
    T_local[1:-1, 0:-2] -
    4 * T_local[1:-1, 1:-1]
  )

  S_internal_local = S_mask_local[1:-1, 1:-1] * Pt

  # Actualizar T_local para el tiempo n+1
  T_local[1:-1, 1:-1] += mu * T_laplacian_local + S_internal_local * dt


# Finalización de la simulación paralela y recolección de datos
comm.Barrier()  # Sincronizar antes de finalizar
end_time = MPI.Wtime() # Detener el temporizador

if rank == 0:
  T_parallel = end_time - start_time
  print(f"{T_parallel:.4f}")