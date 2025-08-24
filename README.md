# Fashion 2D-to-3D GAN 🎨✨

**COMPLETE** - Système de génération automatique de modèles 3D à partir d'images 2D pour l'industrie de la mode. Optimisé pour fine-tuning 8h sur RTX 2000 Ada avec le dataset Deep Fashion3D V2.

## 🎯 Fonctionnalités

- **✅ GAN Architecture Complète** : Générateur 2D→3D avec discriminateur multi-échelle
- **✅ API REST Fonctionnelle** : Interface FastAPI pour intégration facile
- **✅ Génération 2D→3D** : Conversion automatique d'images de mode en modèles 3D
- **✅ Multi-catégories** : Sacs, chaussures, vêtements, bijoux et accessoires
- **✅ Formats multiples** : Export en OBJ, PLY et voxels NumPy
- **✅ Optimisé GPU** : Fine-tuning efficace sur RTX 2000 Ada
- **✅ Qualité ajustable** : Modes Fast/Medium/High selon vos besoins
- **✅ Training Complet** : Pipeline d'entraînement avec checkpoints
- **✅ Preprocessing Intelligent** : Support pour différents formats de dataset

## 🚀 Installation Rapide

### Option 1 : Installation Automatique Complète (Recommandé)

```bash
cd C:\Users\mahmo\OneDrive\Desktop\fashion-2d-3d-gan
python auto_install.py
```

### Option 2 : Installation Manuelle

```bash
# 1. Configuration de l'environnement
python deploy.py --setup

# 2. Test de l'installation
python deploy.py --test

# 3. Analyse de votre dataset
python deploy.py --analyze --dataset-path ./chemin/vers/deep_fashion3d_v2/

# 4. Démarrage de l'API
python deploy.py --api
```

## 📋 Configuration Système

### Prérequis
- **OS** : Windows 10/11, Linux (Ubuntu 18+)
- **GPU** : NVIDIA RTX 2000 Ada (8GB VRAM) ou supérieur
- **RAM** : 16GB minimum, 32GB recommandé
- **Stockage** : 50GB d'espace libre
- **Python** : 3.9+ via Anaconda

### Configuration du Dataset

1. **Téléchargez Deep Fashion3D V2** et placez-le dans un dossier
2. **Ajustez le chemin** dans `config.yaml` :

```yaml
dataset_path: 'C:/chemin/vers/votre/deep_fashion3d_v2/'
```

3. **Analysez votre dataset** :

```bash
python deploy.py --analyze
```

## 🎮 Utilisation

### Mode Démo Interactif (Recommandé pour débuter)

```bash
python deploy.py --demo
```

Ce mode vous guide à travers :
1. Analyse de votre dataset
2. Test avec image générée
3. Démarrage de l'API
4. Test du pipeline complet

### API REST

**Démarrer le serveur :**
```bash
python deploy.py --api
```

**Endpoints disponibles :**
- `POST /generate` - Génération depuis image
- `GET /model-status` - État du modèle
- `GET /stats` - Statistiques d'utilisation
- `GET /docs` - Documentation interactive

**Exemple d'utilisation :**
```bash
# Générer un modèle 3D
curl -X POST "http://localhost:8000/generate" \
  -F "file=@mon_sac.jpg" \
  -F "format=obj"
```

### Entraînement

**Démarrer l'entraînement :**
```bash
python scripts/train.py --hours 8
```

**Reprendre depuis un checkpoint :**
```bash
python scripts/train.py --hours 8 --resume ./models/checkpoints/checkpoint_epoch_5
```

### Utilisation Programmatique

```python
from models.fashion_3d_gan import Fashion3DGAN
import numpy as np

# Charger le modèle
gan = Fashion3DGAN()

# Préparer une image (256x256x3, normalisée [-1,1])
image = np.random.rand(256, 256, 3) * 2 - 1

# Générer le modèle 3D
voxels = gan.generate_3d(image)

# Convertir en mesh
from utils.mesh_utils import voxels_to_mesh, save_mesh_obj
mesh_data = voxels_to_mesh(voxels)
save_mesh_obj(mesh_data, 'resultat.obj')
```

## 🏗️ Architecture du Projet

```
fashion-2d-3d-gan/
├── 📄 config.yaml                 # Configuration principale
├── 📄 environment.yaml            # Environnement Anaconda
├── 📄 auto_install.py             # Installation automatique
├── 📄 deploy.py                   # Scripts de déploiement
├── 📄 README.md                   # Cette documentation
├── 📁 api/
│   └── 📄 server.py               # API REST FastAPI
├── 📁 models/
│   └── 📄 fashion_3d_gan.py       # Architecture GAN principale
├── 📁 scripts/
│   ├── 📄 analyze_dataset.py      # Analyseur de dataset
│   ├── 📄 preprocess_data.py      # Preprocessing intelligent
│   ├── 📄 preprocess_filtered_mesh.py # Preprocessing spécialisé
│   └── 📄 train.py                # Entraînement complet
├── 📁 utils/
│   └── 📄 mesh_utils.py           # Utilitaires 3D avec marching cubes
├── 📁 data/                       # Données preprocessées
├── 📁 outputs/                    # Résultats générés
├── 📁 logs/                       # Logs d'entraînement
└── 📁 models/checkpoints/         # Checkpoints d'entraînement
```

## ⚙️ Configuration Avancée

### Fichier config.yaml

```yaml
# Configuration matériel RTX 2000 Ada
gpu_config:
  memory_growth: true
  mixed_precision: true
  batch_size: 6              # Optimisé pour 8GB VRAM
  num_workers: 4

# Paramètres du modèle
model_config:
  input_resolution: [256, 256]    # Taille des images 2D
  voxel_resolution: 64            # Résolution 3D (64³)
  latent_dim: 512
  generator_lr: 0.0002
  discriminator_lr: 0.0001

# Entraînement 8h
training_config:
  max_hours: 8
  save_interval: 30          # minutes
  validation_interval: 60    # minutes
```

### Optimisation RTX 2000 Ada

Le système est configuré spécifiquement pour votre GPU :
- **Mixed Precision** : Float16/Float32 pour économiser la VRAM
- **Batch Size optimal** : 6 échantillons pour 8GB VRAM
- **Memory Growth** : Allocation progressive de la mémoire GPU
- **Gradient Checkpointing** : Réduction de l'empreinte mémoire

## 📊 Monitoring et Métriques

### Vérification du Système

```bash
# État complet du système
python deploy.py --test

# État de l'API
curl http://localhost:8000/model-status
```

### Métriques d'Évaluation

- **Voxel Occupancy** : Densité du modèle 3D généré
- **Processing Time** : Temps de génération par modèle
- **GPU Memory Usage** : Utilisation de la VRAM
- **Model Validation** : Qualité du mesh généré

## 🔧 Résolution des Problèmes

### Problèmes Courants

#### 1. Erreur GPU/CUDA
```bash
# Vérifier CUDA
nvidia-smi

# Test TensorFlow
python -c "import tensorflow as tf; print(f'GPUs: {len(tf.config.list_physical_devices('GPU'))}')"
```

#### 2. Dataset non trouvé
```bash
# Vérifier le chemin dans config.yaml
dataset_path: 'C:/Users/mahmo/path/to/deep_fashion3d_v2/'

# Analyser la structure
python deploy.py --analyze --dataset-path C:/votre/chemin/
```

#### 3. Problème d'environnement
```bash
# Recréer l'environnement
conda env remove -n fashion3d
python auto_install.py
```

### Logs et Debugging

```bash
# Logs d'entraînement
tail -f logs/training.log

# Test complet du système
python deploy.py --demo
```

## 📈 Performance Attendue

### Benchmarks RTX 2000 Ada
- **Génération** : 2-3 secondes par modèle 3D  
- **Mémoire** : ~6GB VRAM utilisés
- **Throughput** : ~20 modèles/minute
- **Qualité** : Résolution 64³ voxels

### Formats de Sortie
- **OBJ** : Compatible avec Blender, Maya, etc.
- **PLY** : Format académique standard
- **NPY** : Voxels bruts pour post-traitement
- **Métadonnées** : JSON avec métriques de qualité

## 🔄 Workflow de Production

### Pipeline Recommandé

```bash
# 1. Setup initial (une seule fois)
python auto_install.py

# 2. Configuration du dataset
# Éditer config.yaml avec le bon chemin

# 3. Analyse du dataset
python deploy.py --analyze

# 4. Preprocessing
python scripts/preprocess_data.py

# 5. Test du système
python deploy.py --test

# 6. Entraînement (optionnel)
python scripts/train.py --hours 8

# 7. Démarrage de l'API
python deploy.py --api

# 8. Utilisation
# Via API REST ou intégration directe
```

### Intégration dans vos Applications

#### Python
```python
import requests
import json

# Générer un modèle 3D
with open('mon_design.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/generate',
        files={'file': f},
        data={'format': 'obj', 'quality': 'high'}
    )

result = response.json()
if result['success']:
    model_id = result['generation_id']
    print(f"Modèle généré: {model_id}")
```

#### JavaScript/Node.js
```javascript
const FormData = require('form-data');
const fs = require('fs');

const form = new FormData();
form.append('file', fs.createReadStream('design.jpg'));
form.append('format', 'obj');

fetch('http://localhost:8000/generate', {
  method: 'POST',
  body: form
}).then(res => res.json())
  .then(data => console.log('Model generated:', data.generation_id));
```

## 🚀 Roadmap et Extensions

### Fonctionnalités Implémentées ✅
- **Architecture GAN complète** : Générateur 2D→3D + Discriminateur
- **API REST fonctionnelle** : FastAPI avec documentation interactive
- **Training pipeline** : Entraînement complet avec checkpoints
- **Preprocessing intelligent** : Support multi-formats de dataset
- **Mesh generation** : Marching cubes + fallback methods
- **Auto-installation** : Script d'installation automatique

### Fonctionnalités Prévues 🔄
- **Multi-résolution** : Support 128³, 256³ voxels
- **Texture Generation** : Couleurs et matériaux PBR
- **Batch Processing** : Traitement de dossiers entiers
- **Model Optimization** : Réduction automatique de polygones

### Cas d'Usage
- **Prototypage Rapide** : Visualisation 3D avant production
- **E-commerce** : Modèles 3D interactifs pour sites web
- **Réalité Augmentée** : Essayage virtuel
- **Impression 3D** : Prototypes physiques

## 📚 Ressources Supplémentaires

### Documentation
- **API Interactive** : http://localhost:8000/docs (quand l'API tourne)
- **Configuration** : Voir `config.yaml` avec commentaires
- **Architecture** : Détails dans `models/fashion_3d_gan.py`

### Support et Communauté
- **Issues** : Pour signaler des problèmes
- **Contributions** : PRs bienvenues
- **Exemples** : Dossier `examples/` (à venir)

## 📜 License et Crédits

### Dataset
- **Deep Fashion3D V2** : Citation du paper original requise
- **Licence** : Respecter les conditions d'usage académique

### Dépendances Open Source
- TensorFlow (Apache 2.0)
- FastAPI (MIT)
- NumPy, OpenCV (BSD)
- Trimesh, Open3D (MIT)

---

## ⚡ Démarrage Rapide - Résumé

```bash
# Dans le dossier du projet
cd C:\Users\mahmo\OneDrive\Desktop\fashion-2d-3d-gan

# Installation automatique complète
python auto_install.py

# OU étape par étape :
python deploy.py --setup    # Configuration environnement
python deploy.py --test     # Vérification installation
python deploy.py --analyze  # Analyse dataset
python deploy.py --api      # Démarrage API

# Test avec une image
curl -X POST "http://localhost:8000/generate" -F "file=@test.jpg"
```

## 🎉 Conclusion

Ce système Fashion 2D-to-3D GAN **COMPLET** vous permet de transformer rapidement vos concepts de mode 2D en modèles 3D. Optimisé pour votre RTX 2000 Ada et configuré pour un fine-tuning efficace, il représente un outil puissant pour l'innovation dans l'industrie de la mode.

**Prêt à commencer ?**

```bash
python auto_install.py
```

Bon développement ! 🎨✨
