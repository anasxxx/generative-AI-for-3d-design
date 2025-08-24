#!/usr/bin/env python3
"""
Analyseur automatique de structure dataset Deep Fashion3D V2
S'adapte à différentes organisations possibles
"""

import os
import json
from pathlib import Path
from collections import defaultdict, Counter
import yaml
from tqdm import tqdm

class DatasetAnalyzer:
    """Analyseur intelligent de dataset Deep Fashion3D V2"""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.structure = defaultdict(list)
        self.file_stats = defaultdict(int)
        self.extensions = defaultdict(int)
        self.naming_patterns = defaultdict(list)
        
    def analyze_complete_structure(self):
        """Analyse complète de la structure du dataset"""
        print("[INFO] Analyzing Deep Fashion3D V2 structure")
        print(f"[INFO] Path: {self.dataset_path}")
        print("="*60)
        
        if not self.dataset_path.exists():
            print(f"[ERROR] Dataset not found: {self.dataset_path}")
            return None
        
        # 1. Scanner la structure générale
        self.scan_directory_structure()
        
        # 2. Analyser les formats de fichiers
        self.analyze_file_formats()
        
        # 3. Détecter les patterns de nommage
        self.detect_naming_patterns()
        
        # 4. Identifier les appariements image-mesh
        self.find_image_mesh_pairs()
        
        # 5. Analyser les catégories
        self.analyze_categories()
        
        # 6. Générer le rapport
        return self.generate_analysis_report()
    
    def scan_directory_structure(self):
        """Scanner récursivement la structure des dossiers"""
        print("[INFO] Scanning directory structure...")
        
        for root, dirs, files in os.walk(self.dataset_path):
            relative_path = Path(root).relative_to(self.dataset_path)
            level = len(relative_path.parts)
            
            # Enregistrer la structure
            if level <= 3:  # Limiter la profondeur pour la lisibilité
                self.structure[str(relative_path)] = {
                    'dirs': dirs.copy(),
                    'files_count': len(files),
                    'level': level
                }
            
            # Compter les fichiers par extension
            for file in files:
                ext = Path(file).suffix.lower()
                if ext:
                    self.extensions[ext] += 1
                self.file_stats['total_files'] += 1
        
        print(f"[OK] {self.file_stats['total_files']} files found")
    
    def analyze_file_formats(self):
        """Analyser les formats de fichiers présents"""
        print("[INFO] Analyzing file formats...")
        
        # Catégoriser les extensions
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        mesh_exts = {'.obj', '.ply', '.off', '.mesh', '.stl', '.3ds'}
        annotation_exts = {'.json', '.txt', '.xml', '.csv', '.yaml', '.yml'}
        
        self.file_categories = {
            'images': sum(count for ext, count in self.extensions.items() if ext in image_exts),
            'meshes': sum(count for ext, count in self.extensions.items() if ext in mesh_exts),
            'annotations': sum(count for ext, count in self.extensions.items() if ext in annotation_exts),
            'other': sum(count for ext, count in self.extensions.items() 
                        if ext not in (image_exts | mesh_exts | annotation_exts))
        }
        
        print(f"[INFO] Images: {self.file_categories['images']}")
        print(f"[INFO] Meshes: {self.file_categories['meshes']}")
        print(f"[INFO] Annotations: {self.file_categories['annotations']}")
        print(f"[INFO] Other: {self.file_categories['other']}")
    
    def detect_naming_patterns(self):
        """Détecter les patterns de nommage des fichiers"""
        print("[INFO] Detecting naming patterns...")
        
        # Collecter des échantillons de noms
        sample_names = []
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        mesh_exts = {'.obj', '.ply', '.off', '.mesh'}
        
        for root, _, files in os.walk(self.dataset_path):
            for file in files[:100]:  # Limiter pour l'analyse
                stem = Path(file).stem
                ext = Path(file).suffix.lower()
                
                if ext in image_exts:
                    self.naming_patterns['images'].append(stem)
                elif ext in mesh_exts:
                    self.naming_patterns['meshes'].append(stem)
        
        # Analyser les patterns communs
        self.common_patterns = self.analyze_name_patterns()
    
    def analyze_name_patterns(self):
        """Analyser les patterns de nommage pour trouver des correspondances"""
        patterns = {}
        
        # Échantillons d'images et meshes
        img_names = self.naming_patterns['images'][:50]
        mesh_names = self.naming_patterns['meshes'][:50]
        
        if img_names and mesh_names:
            # Chercher des préfixes/suffixes communs
            img_prefixes = set()
            mesh_prefixes = set()
            
            for name in img_names:
                if len(name) > 3:
                    img_prefixes.add(name[:3])
                    img_prefixes.add(name[:5] if len(name) > 5 else name)
            
            for name in mesh_names:
                if len(name) > 3:
                    mesh_prefixes.add(name[:3])
                    mesh_prefixes.add(name[:5] if len(name) > 5 else name)
            
            patterns['common_prefixes'] = list(img_prefixes & mesh_prefixes)
            
            # Chercher des patterns numériques
            import re
            img_numbers = [re.findall(r'\d+', name) for name in img_names]
            mesh_numbers = [re.findall(r'\d+', name) for name in mesh_names]
            
            patterns['has_numbers'] = {
                'images': sum(1 for nums in img_numbers if nums) / len(img_numbers) if img_numbers else 0,
                'meshes': sum(1 for nums in mesh_numbers if nums) / len(mesh_numbers) if mesh_numbers else 0
            }
        
        return patterns
    
    def find_image_mesh_pairs(self):
        """Identifier les appariements potentiels entre images et meshes"""
        print("[INFO] Finding image-mesh pairs...")
        
        # Collecter tous les noms d'images et meshes
        all_images = []
        all_meshes = []
        
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        mesh_exts = {'.obj', '.ply', '.off', '.mesh'}
        
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                stem = Path(file).stem
                ext = Path(file).suffix.lower()
                full_path = os.path.join(root, file)
                
                if ext in image_exts:
                    all_images.append({'stem': stem, 'path': full_path})
                elif ext in mesh_exts:
                    all_meshes.append({'stem': stem, 'path': full_path})
        
        # Essayer différentes stratégies d'appariement
        pairing_strategies = [
            self.exact_match_pairing,
            self.partial_match_pairing,
            self.numeric_match_pairing
        ]
        
        self.pairing_results = {}
        for strategy in pairing_strategies:
            strategy_name = strategy.__name__
            pairs = strategy(all_images[:100], all_meshes[:100])  # Limiter pour l'analyse
            self.pairing_results[strategy_name] = len(pairs)
            print(f"   {strategy_name}: {len(pairs)} pairs found")
    
    def exact_match_pairing(self, images, meshes):
        """Appariement par correspondance exacte des noms"""
        pairs = []
        mesh_dict = {mesh['stem']: mesh for mesh in meshes}
        
        for img in images:
            if img['stem'] in mesh_dict:
                pairs.append((img, mesh_dict[img['stem']]))
        
        return pairs
    
    def partial_match_pairing(self, images, meshes):
        """Appariement par correspondance partielle"""
        pairs = []
        
        for img in images:
            for mesh in meshes:
                # Vérifier si l'un est contenu dans l'autre
                if (img['stem'] in mesh['stem'] or 
                    mesh['stem'] in img['stem'] or
                    self.similarity_score(img['stem'], mesh['stem']) > 0.8):
                    pairs.append((img, mesh))
                    break
        
        return pairs
    
    def numeric_match_pairing(self, images, meshes):
        """Appariement par ID numérique commun"""
        import re
        pairs = []
        
        # Extraire les IDs numériques
        img_ids = {}
        mesh_ids = {}
        
        for img in images:
            numbers = re.findall(r'\d+', img['stem'])
            if numbers:
                img_ids[numbers[0]] = img
        
        for mesh in meshes:
            numbers = re.findall(r'\d+', mesh['stem'])
            if numbers:
                mesh_ids[numbers[0]] = mesh
        
        # Apparier par ID commun
        for id_num in img_ids:
            if id_num in mesh_ids:
                pairs.append((img_ids[id_num], mesh_ids[id_num]))
        
        return pairs
    
    def similarity_score(self, str1, str2):
        """Calculer un score de similarité entre deux chaînes"""
        try:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
        except:
            return 0.0
    
    def analyze_categories(self):
        """Analyser l'organisation des catégories"""
        print("[INFO] Analyzing categories...")
        
        # Chercher des patterns de catégories dans les chemins
        category_indicators = {
            'bag': ['bag', 'handbag', 'purse', 'sac'],
            'shoe': ['shoe', 'boot', 'sneaker', 'chaussure'],
            'top': ['shirt', 'blouse', 'top', 'sweater', 'pull'],
            'bottom': ['pants', 'skirt', 'jeans', 'trouser', 'pantalon'],
            'dress': ['dress', 'gown', 'robe'],
            'accessory': ['hat', 'belt', 'jewelry', 'watch', 'accessoire']
        }
        
        self.detected_categories = defaultdict(int)
        
        for path_key in self.structure.keys():
            path_lower = path_key.lower()
            for category, indicators in category_indicators.items():
                for indicator in indicators:
                    if indicator in path_lower:
                        self.detected_categories[category] += 1
                        break
        
        print(f"[INFO] Categories detected: {dict(self.detected_categories)}")
    
    def generate_analysis_report(self):
        """Générer un rapport complet d'analyse"""
        report = {
            'dataset_path': str(self.dataset_path),
            'total_files': self.file_stats['total_files'],
            'directory_structure': dict(self.structure),
            'file_extensions': dict(self.extensions),
            'file_categories': self.file_categories,
            'naming_patterns': dict(self.naming_patterns),
            'common_patterns': self.common_patterns,
            'pairing_results': self.pairing_results,
            'detected_categories': dict(self.detected_categories),
            'recommendations': self.generate_recommendations()
        }
        
        # Sauvegarder le rapport
        report_path = Path('./dataset_analysis_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n[INFO] Report saved: {report_path}")
        
        # Afficher un résumé
        self.print_summary_report(report)
        
        return report
    
    def generate_recommendations(self):
        """Générer des recommandations pour le preprocessing"""
        recommendations = []
        
        # Recommandations basées sur les appariements
        if self.pairing_results:
            best_pairing = max(self.pairing_results.items(), key=lambda x: x[1])
            recommendations.append(f"Use {best_pairing[0]} for pairing ({best_pairing[1]} pairs)")
        
        # Recommandations sur les formats
        if self.extensions.get('.obj', 0) > 0:
            recommendations.append("Format .obj detected - compatible with system")
        elif self.extensions.get('.ply', 0) > 0:
            recommendations.append("Format .ply detected - compatible with system")
        else:
            recommendations.append("[WARNING] Non-standard mesh formats detected")
        
        # Recommandations sur la taille du dataset
        if self.file_categories['images'] > 10000:
            recommendations.append("Large dataset detected - use sampling for 8h fine-tuning")
        elif self.file_categories['images'] < 1000:
            recommendations.append("[WARNING] Small dataset - consider data augmentation")
        
        return recommendations
    
    def print_summary_report(self, report):
        """Afficher un résumé du rapport"""
        print("\n" + "="*60)
        print("[INFO] DATASET ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Path: {report['dataset_path']}")
        print(f"Total files: {report['total_files']:,}")
        print()
        
        print("File categories:")
        for category, count in report['file_categories'].items():
            print(f"   {category}: {count:,}")
        print()
        
        print("Pairing possibilities:")
        for strategy, count in report['pairing_results'].items():
            print(f"   {strategy}: {count}")
        print()
        
        print("Fashion categories detected:")
        for category, count in report['detected_categories'].items():
            print(f"   {category}: {count}")
        print()
        
        print("Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "="*60)

def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deep Fashion3D V2 Dataset Analyzer')
    parser.add_argument('--path', default='./deep_fashion3d_v2/', 
                       help='Path to dataset')
    parser.add_argument('--output', default='./dataset_analysis_report.json',
                       help='Output report file')
    
    args = parser.parse_args()
    
    # Lancer l'analyse
    analyzer = DatasetAnalyzer(args.path)
    report = analyzer.analyze_complete_structure()
    
    if report:
        print(f"\n[OK] Analysis completed!")
        print(f"[INFO] Detailed report: {args.output}")
        
        # Mettre à jour la configuration si nécessaire
        if Path('config.yaml').exists():
            try:
                with open('config.yaml', 'r') as f:
                    config = yaml.safe_load(f)
                
                # Mettre à jour le chemin du dataset
                config['dataset_path'] = str(analyzer.dataset_path)
                
                with open('config.yaml', 'w') as f:
                    yaml.dump(config, f)
                
                print("[INFO] Configuration updated with dataset path")
            except Exception as e:
                print(f"[WARNING] Could not update config: {e}")

if __name__ == "__main__":
    main()
