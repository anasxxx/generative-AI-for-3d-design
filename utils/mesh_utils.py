#!/usr/bin/env python3
"""
Utilitaires pour la conversion et manipulation de meshes 3D
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path

def voxels_to_mesh(voxels: np.ndarray, threshold: float = 0.5, smooth: bool = True):
    """
    Convertir un array de voxels en mesh 3D
    
    Args:
        voxels: Array 3D de voxels (64x64x64)
        threshold: Seuil pour la surface (0.5 par défaut)
        smooth: Appliquer un lissage au mesh
    
    Returns:
        Mesh data dictionary
    """
    try:
        # Ensure voxels are in the right format
        if len(voxels.shape) != 3:
            raise ValueError(f"Expected 3D array, got shape {voxels.shape}")
        
        # Normalize voxels to [0, 1] if needed
        if voxels.max() > 1.0 or voxels.min() < 0.0:
            voxels = np.clip(voxels, 0, 1)
        
        print(f"[INFO] Converting voxels {voxels.shape} to mesh")
        
        # Try to use marching cubes if available
        try:
            import mcubes
            vertices, faces = mcubes.marching_cubes(voxels, threshold)
            
            # Scale vertices to [-1, 1] range
            vertices = (vertices / (voxels.shape[0] - 1)) * 2 - 1
            
            mesh_data = {
                'vertices': vertices,
                'faces': faces,
                'is_valid': len(vertices) > 0 and len(faces) > 0,
                'vertex_count': len(vertices),
                'face_count': len(faces)
            }
            
            print(f"[OK] Marching cubes generated {len(vertices)} vertices, {len(faces)} faces")
            return mesh_data
            
        except ImportError:
            print("[WARNING] mcubes not available, using fallback method")
            return create_mesh_from_voxels_fallback(voxels, threshold)
        
    except Exception as e:
        print(f"[ERROR] Voxel to mesh conversion error: {e}")
        return create_fallback_mesh()

def create_mesh_from_voxels_fallback(voxels: np.ndarray, threshold: float = 0.5):
    """Fallback mesh generation when marching cubes is not available"""
    print("[INFO] Using fallback mesh generation...")
    
    # Find occupied voxels
    occupied = voxels > threshold
    
    if not np.any(occupied):
        print("[WARNING] No occupied voxels found, creating empty mesh")
        return create_fallback_mesh()
    
    # Get occupied positions
    occupied_positions = np.where(occupied)
    
    # Create vertices from occupied voxels
    vertices = []
    faces = []
    
    # Scale to [-1, 1] range
    scale_factor = 2.0 / (voxels.shape[0] - 1)
    
    for i in range(len(occupied_positions[0])):
        x, y, z = occupied_positions[0][i], occupied_positions[1][i], occupied_positions[2][i]
        
        # Convert to [-1, 1] range
        x_norm = x * scale_factor - 1.0
        y_norm = y * scale_factor - 1.0
        z_norm = z * scale_factor - 1.0
        
        vertices.append([x_norm, y_norm, z_norm])
    
    # Create simple faces (triangles) between adjacent vertices
    # This is a simplified approach - in practice you'd want proper surface reconstruction
    if len(vertices) > 3:
        # Create some basic triangular faces
        for i in range(0, len(vertices) - 2, 3):
            if i + 2 < len(vertices):
                faces.append([i, i + 1, i + 2])
    
    vertices = np.array(vertices)
    faces = np.array(faces) if faces else np.array([[0, 1, 2]])
    
    mesh_data = {
        'vertices': vertices,
        'faces': faces,
        'is_valid': len(vertices) > 0,
        'vertex_count': len(vertices),
        'face_count': len(faces)
    }
    
    print(f"[OK] Fallback method generated {len(vertices)} vertices, {len(faces)} faces")
    return mesh_data

def create_fallback_mesh():
    """Créer un mesh de secours (cube simple)"""
    # Vertices d'un cube simple
    vertices = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
    ])
    
    # Faces du cube
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 7, 6], [4, 6, 5],
        [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]
    ])
    
    return {
        'vertices': vertices,
        'faces': faces,
        'is_valid': True,
        'vertex_count': len(vertices),
        'face_count': len(faces)
    }

def save_mesh_obj(mesh_data: dict, filename: str) -> bool:
    """
    Sauvegarder un mesh au format OBJ
    """
    try:
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        
        with open(filename, 'w') as f:
            f.write("# Fashion 2D-to-3D GAN Generated Mesh\n")
            f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")
            
            # Écrire les vertices
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            f.write("\n")
            
            # Écrire les faces (OBJ utilise des indices 1-based)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        print(f"[OK] Mesh saved to {filename}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save OBJ: {e}")
        return False

def save_mesh_ply(mesh_data: dict, filename: str) -> bool:
    """
    Sauvegarder un mesh au format PLY
    """
    try:
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Écrire les vertices
            for vertex in vertices:
                f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Écrire les faces
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        print(f"[OK] PLY mesh saved to {filename}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save PLY: {e}")
        return False

def save_mesh_formats(mesh_data: dict, base_path: str, formats: list = ['obj', 'ply']) -> dict:
    """
    Sauvegarder un mesh dans plusieurs formats
    
    Args:
        mesh_data: Données du mesh
        base_path: Chemin de base (sans extension)
        formats: Liste des formats à sauvegarder
    
    Returns:
        Dict des chemins sauvegardés
    """
    saved_paths = {}
    base_path = Path(base_path)
    
    for format in formats:
        try:
            file_path = base_path.with_suffix(f'.{format}')
            
            if format == 'obj':
                success = save_mesh_obj(mesh_data, str(file_path))
            elif format == 'ply':
                success = save_mesh_ply(mesh_data, str(file_path))
            else:
                print(f"[WARNING] Unsupported format: {format}")
                continue
            
            if success:
                saved_paths[format] = str(file_path)
            
        except Exception as e:
            print(f"[ERROR] Failed to save {format}: {e}")
    
    return saved_paths

def validate_mesh(mesh_data: dict) -> dict:
    """
    Valider et analyser un mesh 3D
    
    Returns:
        Dict avec les métriques de validation
    """
    metrics = {
        'is_valid': mesh_data.get('is_valid', False),
        'num_vertices': len(mesh_data.get('vertices', [])),
        'num_faces': len(mesh_data.get('faces', [])),
        'bounds': None
    }
    
    try:
        vertices = mesh_data.get('vertices')
        if vertices is not None and len(vertices) > 0:
            vertices = np.array(vertices)
            metrics['bounds'] = {
                'min': vertices.min(axis=0).tolist(),
                'max': vertices.max(axis=0).tolist()
            }
        
    except Exception as e:
        print(f"[WARNING] Error calculating metrics: {e}")
    
    return metrics

def optimize_mesh(mesh_data: dict, target_faces: int = 2000) -> dict:
    """
    Optimiser un mesh pour réduire le nombre de faces (version basique)
    """
    try:
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        
        if len(faces) <= target_faces:
            return mesh_data
        
        # Simplification basique : prendre un échantillon aléatoire des faces
        # En production, utiliserait des algorithmes plus sophistiqués
        indices = np.random.choice(len(faces), target_faces, replace=False)
        optimized_faces = faces[indices]
        
        # Identifier les vertices utilisés
        used_vertices = np.unique(optimized_faces.flatten())
        new_vertices = vertices[used_vertices]
        
        # Remap les indices des faces
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        new_faces = np.array([[vertex_map[face[i]] for i in range(3)] for face in optimized_faces])
        
        optimized_mesh = {
            'vertices': new_vertices,
            'faces': new_faces,
            'is_valid': True,
            'vertex_count': len(new_vertices),
            'face_count': len(new_faces)
        }
        
        print(f"[INFO] Mesh optimized: {len(faces)} -> {len(new_faces)} faces")
        return optimized_mesh
        
    except Exception as e:
        print(f"[WARNING] Optimization error: {e}")
        return mesh_data

def estimate_processing_time(voxels: np.ndarray, quality: str) -> float:
    """Estimer le temps de traitement basé sur la taille et la qualité"""
    base_time = voxels.size / 1000000  # Temps de base basé sur la taille
    
    quality_multipliers = {
        'fast': 1.0,
        'medium': 2.5,
        'high': 5.0
    }
    
    return base_time * quality_multipliers.get(quality, 2.5)

def get_supported_formats() -> list:
    """Retourner la liste des formats supportés"""
    return ['obj', 'ply', 'npy']

def validate_voxel_input(voxels: np.ndarray) -> bool:
    """Valider les voxels d'entrée"""
    if len(voxels.shape) != 3:
        return False
    
    if not (32 <= voxels.shape[0] <= 128):  # Résolution raisonnable
        return False
    
    if not (0 <= voxels.min() and voxels.max() <= 1):  # Valeurs dans [0,1]
        return False
    
    return True

class MeshProcessor:
    """Classe pour le traitement avancé de meshes"""
    
    def __init__(self, quality: str = 'high'):
        self.quality = quality
        
        # Paramètres selon la qualité
        self.params = {
            'fast': {
                'target_faces': 500,
                'smooth_iterations': 0
            },
            'medium': {
                'target_faces': 1000,
                'smooth_iterations': 1
            },
            'high': {
                'target_faces': 2000,
                'smooth_iterations': 2
            }
        }
    
    def process_voxels_to_mesh(self, voxels: np.ndarray) -> dict:
        """
        Pipeline complet de traitement voxels -> mesh
        
        Returns:
            Dict avec le mesh et les métadonnées
        """
        # 1. Conversion voxels -> mesh
        mesh_data = voxels_to_mesh(voxels, smooth=True)
        
        # 2. Optimisation selon la qualité
        mesh_data = optimize_mesh(mesh_data, self.params[self.quality]['target_faces'])
        
        # 3. Validation
        validation = validate_mesh(mesh_data)
        
        return {
            'mesh': mesh_data,
            'validation': validation,
            'processing_quality': self.quality,
            'face_count': mesh_data['face_count'],
            'vertex_count': mesh_data['vertex_count']
        }
    
    def create_complete_output(self, voxels: np.ndarray, output_dir: str, name: str) -> dict:
        """
        Créer un output complet avec mesh et métadonnées
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Traitement du mesh
        result = self.process_voxels_to_mesh(voxels)
        mesh_data = result['mesh']
        
        # 2. Sauvegarde dans différents formats
        base_path = output_dir / name
        saved_formats = save_mesh_formats(mesh_data, str(base_path), ['obj', 'ply'])
        
        # 3. Sauvegarder les voxels originaux
        voxel_path = output_dir / f'{name}_voxels.npy'
        np.save(voxel_path, voxels)
        
        # 4. Créer les métadonnées
        metadata = {
            'generation_info': {
                'name': name,
                'timestamp': str(np.datetime64('now')),
                'voxel_resolution': voxels.shape[0],
                'processing_quality': self.quality
            },
            'mesh_info': result['validation'],
            'files': {
                'voxels': str(voxel_path),
                'formats': saved_formats
            },
            'statistics': {
                'voxel_occupancy': float(np.sum(voxels > 0.5) / voxels.size),
                'mesh_face_count': result['face_count'],
                'mesh_vertex_count': result['vertex_count']
            }
        }
        
        # 5. Sauvegarder les métadonnées
        metadata_path = output_dir / f'{name}_metadata.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return metadata

# Fonctions utilitaires pour l'API
def quick_voxel_to_obj(voxels: np.ndarray, output_path: str) -> str:
    """Conversion rapide voxels -> OBJ pour l'API"""
    mesh_data = voxels_to_mesh(voxels, threshold=0.5, smooth=False)
    save_mesh_obj(mesh_data, output_path)
    return output_path

# Test des utilitaires
if __name__ == "__main__":
    print("[INFO] Testing mesh utilities...")
    
    # Créer des voxels de test
    test_voxels = np.random.rand(32, 32, 32)
    test_voxels = (test_voxels > 0.7).astype(np.float32)
    
    # Test du processeur
    processor = MeshProcessor('high')
    result = processor.process_voxels_to_mesh(test_voxels)
    
    print(f"[OK] Test completed - {result['face_count']} faces generated")
    
    # Test de sauvegarde
    mesh_data = result['mesh']
    save_mesh_obj(mesh_data, 'test_output.obj')
    
    print("[OK] Mesh utilities ready!")
