import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv
from datetime import datetime
from filterpy.kalman import KalmanFilter
from scipy.spatial.transform import Rotation as R

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

# Función de matriz de transformación homogénea con datos filtrados
def get_homogeneous_matrix(position, quaternion):
    r = R.from_quat(quaternion)
    rotation_matrix = r.as_matrix()
    # Normalizar la matriz de rotación para asegurar ortonormalidad
    u, _, vh = np.linalg.svd(rotation_matrix)
    rotation_matrix = np.dot(u, vh)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position
    return transformation_matrix

# Generar todas las transformaciones utilizando los datos filtrados
def multiply_transformations(filtered_positions, orientations, frame_idx):
    # Definir una transformación inicial estándar (posición anatómica neutra)
    initial_transformation = np.eye(4)
    initial_transformation[:3, 3] = [0, 0, 0]  # Punto de referencia global
    final_transformation = initial_transformation
    transformations = []
    for joint in jointNames:
        position = [filtered_positions[joint]['x'][frame_idx],
                    filtered_positions[joint]['y'][frame_idx],
                    filtered_positions[joint]['z'][frame_idx]]
        orientation = orientations[joint][frame_idx]
        transformation_matrix = get_homogeneous_matrix(position, orientation)
        final_transformation = np.dot(final_transformation, transformation_matrix)
        transformations.append(final_transformation)
    return transformations

# Calcular todas las transformaciones para cada frame
all_transformations = [multiply_transformations(filtered_positions, orientations, idx) 
                       for idx in range(len(timestamps))]

# Obtener los ángulos articulares en función de las transformaciones homogéneas filtradas
# y restringir los ángulos al rango fisiológico de cada articulación
def get_joint_angles_for_all_frames(all_transformations):
    joint_limits = {
        'ShoulderLeft': {'flexion': (-45, 180), 'abduction': (0, 180), 'rotation': (-60, 60)},
        'ElbowLeft': {'flexion': (0, 150)},
        'ShoulderRight': {'flexion': (-45, 180), 'abduction': (0, 180), 'rotation': (-60, 60)},
        'ElbowRight': {'flexion': (0, 150)}
    }
    
    all_joint_angles = []
    for frame_transformations in all_transformations:
        joint_angles = {}
        for joint_idx, transformation in enumerate(frame_transformations):
            rotation_matrix = transformation[:3, :3]
            r = R.from_matrix(rotation_matrix)
            euler_angles = r.as_euler('xyz', degrees=True)
            
            # Restringir los ángulos al rango fisiológico
            joint_name = jointNames[joint_idx]
            if joint_name in joint_limits:
                limits = joint_limits[joint_name]
                if 'flexion' in limits:
                    euler_angles[0] = np.clip(euler_angles[0], limits['flexion'][0], limits['flexion'][1])
                if 'abduction' in limits:
                    euler_angles[1] = np.clip(euler_angles[1], limits['abduction'][0], limits['abduction'][1])
                if 'rotation' in limits:
                    euler_angles[2] = np.clip(euler_angles[2], limits['rotation'][0], limits['rotation'][1])
            
            # Restringir los ángulos a -180 a 180 grados
            euler_angles = np.mod(euler_angles + 180, 360) - 180
            joint_angles[joint_name] = euler_angles
        all_joint_angles.append(joint_angles)
    return all_joint_angles

all_joint_angles = get_joint_angles_for_all_frames(all_transformations)

# Calcular la distancia promedio entre articulaciones
def calculate_average_distances(filtered_positions, joint_pairs):
    average_distances = {}
    for joint1, joint2 in joint_pairs:
        dist_sum = 0
        num_frames = len(filtered_positions[joint1]['x'])
        
        for i in range(num_frames):
            pos1 = np.array([filtered_positions[joint1]['x'][i], filtered_positions[joint1]['y'][i], filtered_positions[joint1]['z'][i]])
            pos2 = np.array([filtered_positions[joint2]['x'][i], filtered_positions[joint2]['y'][i], filtered_positions[joint2]['z'][i]])
            dist_sum += np.linalg.norm(pos2 - pos1)
        
        average_distances[(joint1, joint2)] = dist_sum / num_frames
    return average_distances

joint_pairs = [('ClavicleLeft', 'ShoulderLeft'), 
               ('ShoulderLeft', 'ElbowLeft'), 
               ('ElbowLeft', 'WristLeft'), 
               ('WristLeft', 'HandLeft')]

average_distances = calculate_average_distances(filtered_positions, joint_pairs)

# Aplicar la cinemática directa usando los ángulos filtrados y distancias promedio
def forward_kinematics(joint_angles, average_distances):
    base_position = np.array([0, 0, 0])
    joint_positions = {'ClavicleLeft': base_position}
    joints_sequence = ['ClavicleLeft', 'ShoulderLeft', 'ElbowLeft', 'WristLeft', 'HandLeft']
    
    for i in range(1, len(joints_sequence)):
        joint1 = joints_sequence[i-1]
        joint2 = joints_sequence[i]
        
        angles = joint_angles[joint1]
        rot = R.from_euler('xyz', angles, degrees=True)
        distance = average_distances[(joint1, joint2)]
        local_displacement = np.array([0, 0, distance])
        global_displacement = rot.apply(local_displacement)
        joint_positions[joint2] = joint_positions[joint1] + global_displacement
    
    return joint_positions

calculated_positions = [forward_kinematics(all_joint_angles[frame], average_distances) 
                        for frame in range(len(all_joint_angles))]

# Visualización comparativa de posiciones originales vs. filtradas
def visualize_filtered_vs_original(positions, filtered_positions, jointNames):
    for joint in jointNames:
        time_points = range(len(timestamps))
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(time_points, positions[joint]['x'], label='Original X', alpha=0.6)
        plt.plot(time_points, filtered_positions[joint]['x'], label='Filtrado X', linestyle='--')
        plt.xlabel('Frame')
        plt.ylabel('Posición X (mm)')
        plt.legend()
        plt.title(f'{joint} - Eje X')
        
        plt.subplot(1, 3, 2)
        plt.plot(time_points, positions[joint]['y'], label='Original Y', alpha=0.6)
        plt.plot(time_points, filtered_positions[joint]['y'], label='Filtrado Y', linestyle='--')
        plt.xlabel('Frame')
        plt.ylabel('Posición Y (mm)')
        plt.legend()
        plt.title(f'{joint} - Eje Y')
        
        plt.subplot(1, 3, 3)
        plt.plot(time_points, positions[joint]['z'], label='Original Z', alpha=0.6)
        plt.plot(time_points, filtered_positions[joint]['z'], label='Filtrado Z', linestyle='--')
        plt.xlabel('Frame')
        plt.ylabel('Posición Z (mm)')
        plt.legend()
        plt.title(f'{joint} - Eje Z')
        
        plt.tight_layout()
        plt.show()

# Llamada a la función de visualización
visualize_filtered_vs_original(positions, filtered_positions, jointNames)

# Exportar los ángulos articulares para OpenSim en un archivo .mot
shoulder_angles = [all_joint_angles[i]['ShoulderRight'] for i in range(len(timestamps))]
elbow_angles = [all_joint_angles[i]['ElbowRight'] for i in range(len(timestamps))]

# Crear DataFrame para el archivo .mot
data = {
    'time': np.linspace(0, len(timestamps) * 0.01, len(timestamps)),  # Ajustar intervalo de tiempo
    'r_shoulder_elev': [angles[0] for angles in shoulder_angles],  # Ángulo en elevación del hombro
    'r_elbow_flex': [angles[0] for angles in elbow_angles]       # Flexión del codo
}

mot_df = pd.DataFrame(data)

# Configuración del archivo .mot
mot_filename = 'output_motion_Angulos.mot'
with open(mot_filename, 'w') as f:
    # Escribir encabezado para OpenSim
    f.write("Coordinates\n")
    f.write(f"nRows={len(mot_df)}\n")
    f.write("nColumns=3\n")
    f.write("Units are S.I. units (second, meters, Newtons, ...)\n")
    f.write("Angles are in degrees.\n")
    f.write("endheader\n")
    mot_df.to_csv(f, sep='\t', index=False)

print(f"Archivo .mot exportado como {mot_filename}")