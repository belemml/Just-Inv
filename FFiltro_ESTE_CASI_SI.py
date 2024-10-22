import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv
from datetime import datetime
from filterpy.kalman import KalmanFilter
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

# Cargar y procesar el archivo JSON
filename = 'skeleton_data_2024-08-28T16-49-35.json'
with open(filename, 'r') as f:
    jsonData = json.load(f)

# Definir nombres de articulaciones y diccionarios para datos filtrados
jointNames = ['SpineChest', 'Neck', 'ClavicleLeft', 'ShoulderLeft', 'ElbowLeft', 
              'WristLeft', 'HandLeft', 'HandTipLeft', 'ClavicleRight', 
              'ShoulderRight', 'ElbowRight', 'WristRight', 'HandRight', 'HandTipRight']

positions = {joint: {'x': [], 'y': [], 'z': []} for joint in jointNames}
orientations = {joint: [] for joint in jointNames}
timestamps = []

# Extraer y almacenar datos de posiciones y orientaciones
for frame in jsonData:
    timestamps.append(datetime.fromisoformat(frame['timestamp']))  
    joints = frame['joints']
    for jointName in jointNames:
        jointData = next((joint for joint in joints if joint['name'] == jointName), None)
        if jointData:
            positions[jointName]['x'].append(jointData['position']['x'])
            positions[jointName]['y'].append(jointData['position']['y'])
            positions[jointName]['z'].append(jointData['position']['z'])
            orientations[jointName].append([jointData['orientation']['x'], 
                                            jointData['orientation']['y'], 
                                            jointData['orientation']['z'], 
                                            jointData['orientation']['w']])

# Función para aplicar el filtro de Kalman con ajustes en R y Q
def apply_kalman_filter(data):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([data[0], 0])  # Estado inicial
    kf.F = np.array([[1, 1], [0, 1]])  # Matriz de transición
    kf.H = np.array([[1, 0]])  # Matriz de medida
    kf.P *= 1000.0  # Incertidumbre inicial
    kf.R = 15  # Ruido de medida mayor para suavizar más
    kf.Q = np.array([[2, 0], [0, 2]])  # Ruido de proceso ajustado para más flexibilidad

    filtered_data = []
    for z in data:
        kf.predict()
        kf.update(z)
        filtered_data.append(kf.x[0])
    return filtered_data

# Aplicación del filtro de Kalman con estos nuevos valores
filtered_positions = {joint: {} for joint in jointNames}
for joint in jointNames:
    filtered_positions[joint]['x'] = apply_kalman_filter(positions[joint]['x'])
    filtered_positions[joint]['y'] = apply_kalman_filter(positions[joint]['y'])
    filtered_positions[joint]['z'] = apply_kalman_filter(positions[joint]['z'])

# Definir pares de articulaciones para la cinemática inversa
joint_pairs = [('ClavicleLeft', 'ShoulderLeft'), 
               ('ShoulderLeft', 'ElbowLeft'), 
               ('ElbowLeft', 'WristLeft'), 
               ('ClavicleRight', 'ShoulderRight'), 
               ('ShoulderRight', 'ElbowRight'), 
               ('ElbowRight', 'WristRight')]

# Función para obtener la distancia entre dos articulaciones
def joint_distance(joint1, joint2, frame_idx, positions):
    pos1 = np.array([positions[joint1]['x'][frame_idx],
                     positions[joint1]['y'][frame_idx],
                     positions[joint1]['z'][frame_idx]])
    pos2 = np.array([positions[joint2]['x'][frame_idx],
                     positions[joint2]['y'][frame_idx],
                     positions[joint2]['z'][frame_idx]])
    return np.linalg.norm(pos2 - pos1)

# Cálculo de los ángulos mediante cinemática inversa
def inverse_kinematics(target_position, initial_guess, joint1, joint2, frame_idx, positions):
    def objective_function(angles):
        rot = R.from_euler('xyz', angles, degrees=True)
        local_displacement = np.array([0, 0, joint_distance(joint1, joint2, frame_idx, positions)])
        estimated_position = rot.apply(local_displacement)
        current_position = np.array([positions[joint1]['x'][frame_idx],
                                     positions[joint1]['y'][frame_idx],
                                     positions[joint1]['z'][frame_idx]])
        return np.linalg.norm(current_position + estimated_position - target_position)

    result = minimize(objective_function, initial_guess, method='BFGS')
    return result.x

# Cálculo de todos los ángulos para cada frame
all_joint_angles_ik = []
initial_guess = [0, 0, 0]  # Suposición inicial para los ángulos (en grados)
for frame_idx in range(len(timestamps)):
    joint_angles = {}
    for joint1, joint2 in joint_pairs:
        target_position = np.array([filtered_positions[joint2]['x'][frame_idx],
                                    filtered_positions[joint2]['y'][frame_idx],
                                    filtered_positions[joint2]['z'][frame_idx]])
        angles = inverse_kinematics(target_position, initial_guess, joint1, joint2, frame_idx, filtered_positions)
        # Restringir los ángulos al rango anatómico permitido
        if joint2 == 'ShoulderLeft' or joint2 == 'ShoulderRight':
            angles[0] = np.clip(angles[0], -45, 180)  # Flexión/extensión
            angles[1] = np.clip(angles[1], -30, 180)  # Abducción/aducción
            angles[2] = np.clip(angles[2], -60, 60)   # Rotación interna/externa
        elif joint2 == 'ElbowLeft' or joint2 == 'ElbowRight':
            angles[0] = np.clip(angles[0], 0, 150)    # Flexión del codo
        joint_angles[joint2] = angles
    all_joint_angles_ik.append(joint_angles)

# Exportar los ángulos articulares calculados por cinemática inversa para OpenSim en un archivo .mot
shoulder_angles_ik = [all_joint_angles_ik[i]['ShoulderRight'] for i in range(len(timestamps))]
elbow_angles_ik = [all_joint_angles_ik[i]['ElbowRight'] for i in range(len(timestamps))]

# Crear DataFrame para el archivo .mot
data_ik = {
    'time': np.linspace(0, len(timestamps) * 0.01, len(timestamps)),  # Ajustar intervalo de tiempo
    'r_shoulder_elev': [angles[0] for angles in shoulder_angles_ik],  # Ángulo en elevación del hombro
    'r_elbow_flex': [angles[0] for angles in elbow_angles_ik]         # Flexión del codo
}

mot_df_ik = pd.DataFrame(data_ik)

# Configuración del archivo .mot
mot_filename_ik = 'output_motion_ik.mot'
with open(mot_filename_ik, 'w') as f:
    # Escribir encabezado para OpenSim
    f.write("Coordinates\n")
    f.write(f"nRows={len(mot_df_ik)}\n")
    f.write("nColumns=3\n")
    f.write("Units are S.I. units (second, meters, Newtons, ...)\n")
    f.write("Angles are in degrees.\n")
    f.write("endheader\n")
    mot_df_ik.to_csv(f, sep='\t', index=False)

print(f"Archivo .mot exportado como {mot_filename_ik}")

# Visualización comparativa de posiciones filtradas vs. sin filtrar
def plot_filtered_vs_unfiltered_all_axes(joint_name):
    time_points = range(len(timestamps))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(time_points, positions[joint_name]['x'], label='Sin filtrar (x)', color='blue', alpha=0.6)
    axes[0].plot(time_points, filtered_positions[joint_name]['x'], label='Filtrado (x)', color='red', linestyle='--')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Posición X (mm)')
    axes[0].set_title(f'Comparación de señal filtrada vs. sin filtrar para {joint_name} - X')
    axes[0].legend()

    axes[1].plot(time_points, positions[joint_name]['y'], label='Sin filtrar (y)', color='blue', alpha=0.6)
    axes[1].plot(time_points, filtered_positions[joint_name]['y'], label='Filtrado (y)', color='red', linestyle='--')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Posición Y (mm)')
    axes[1].set_title(f'Comparación de señal filtrada vs. sin filtrar para {joint_name} - Y')
    axes[1].legend()

    axes[2].plot(time_points, positions[joint_name]['z'], label='Sin filtrar (z)', color='blue', alpha=0.6)
    axes[2].plot(time_points, filtered_positions[joint_name]['z'], label='Filtrado (z)', color='orange', linestyle='--')
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Posición Z (mm)')
    axes[2].set_title(f'Comparación de señal filtrada vs. sin filtrar para {joint_name} - Z')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

# Graficar la comparación para todas las articulaciones y todos los ejes
for joint in jointNames:
    plot_filtered_vs_unfiltered_all_axes(joint)