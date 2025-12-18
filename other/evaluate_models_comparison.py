#!/usr/bin/env python3
"""
Comprehensive Model Evaluation and Comparison Script

Tests both Custom YOLOv8x-world and Standard YOLOv8x models on test datasets:
- outlet_test_far (all objects should be 'outlet')
- socket_test_close (all objects should be 'outlet')  
- fluor_lamp_test (all objects should be 'fluorescent tube')
- fire_extinguisher_test (all objects should be 'fire extinguisher')

Calculates and compares:
- Accuracy per dataset
- F1-score per dataset
- Average accuracy and F1-score
- Detailed performance comparison between models
"""

import os
import cv2
import glob
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import json


class ModelEvaluator:
    """Evaluates object detection model performance on test datasets."""
    
    # Define synonyms for each object class
    CLASS_SYNONYMS = {
        'outlet': [
            'outlet', 'socket', 'power outlet', 'wall outlet', 'electrical outlet',
            'power socket', 'wall socket', 'plug socket', 'receptacle', 'power point'
        ],
        'fluorescent tube': [
            'fluorescent tube', 'fluorescent lamp', 'fluorescent light', 'tube light',
            'fluorescent', 'lamp', 'light tube', 'ceiling light', 'tube lamp'
        ],
        'fire extinguisher': [
            'fire extinguisher', 'extinguisher', 'emergency extinguisher', 
            'fire suppressor', 'fire safety equipment', 'safety extinguisher'
        ]
    }
    
    def __init__(self, model_path, model_name, use_custom_classes=False):
        """
        Initialize evaluator with model.
        
        Args:
            model_path: Path to model weights (.pt file)
            model_name: Name of the model for reporting
            use_custom_classes: Whether to configure model with custom synonym classes
        """
        self.model_path = model_path
        self.model_name = model_name
        self.use_custom_classes = use_custom_classes
        self.model = None
        self.results = {
            'model_name': model_name,
            'model_path': model_path,
            'datasets': {},
            'overall': {}
        }
    
    def _is_class_match(self, detected_class, expected_class):
        """
        Check if detected class matches expected class using synonym matching.
        
        Args:
            detected_class: The class name detected by the model
            expected_class: The expected class name for the dataset
            
        Returns:
            bool: True if classes match (including synonyms), False otherwise
        """
        detected_lower = detected_class.lower().strip()
        expected_lower = expected_class.lower().strip()
        
        # Direct match
        if detected_lower == expected_lower:
            return True
        
        # Check synonyms
        for canonical_class, synonyms in self.CLASS_SYNONYMS.items():
            # Normalize synonyms to lowercase
            synonyms_lower = [s.lower().strip() for s in synonyms]
            
            # Check if both detected and expected are in the same synonym group
            if expected_lower in synonyms_lower and detected_lower in synonyms_lower:
                return True
        
        return False
        
    def load_model(self):
        """Load the YOLO model."""
        print(f"\n{'='*70}")
        print(f"Loading Model: {self.model_name}")
        print(f"Path: {self.model_path}")
        print(f"{'='*70}")
        
        # YOLO will auto-download standard models if they don't exist
        self.model = YOLO(self.model_path)
        
        # Only configure custom classes if requested (for standard model)
        if self.use_custom_classes:
            try:
                # Combine all synonym lists into one master list
                all_classes = []
                for synonyms in self.CLASS_SYNONYMS.values():
                    all_classes.extend(synonyms)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_classes = []
                for cls in all_classes:
                    if cls not in seen:
                        seen.add(cls)
                        unique_classes.append(cls)
                
                self.model.set_classes(unique_classes)
                print(f"✅ Model loaded successfully!")
                print(f"   🔍 YOLO-World model configured with {len(unique_classes)} class variations")
                print(f"   📋 Classes: {', '.join(unique_classes[:10])}{'...' if len(unique_classes) > 10 else ''}")
            except AttributeError:
                # Standard YOLO models don't have set_classes method
                print(f"✅ Model loaded successfully!")
                print(f"   ℹ️  Standard YOLO model (uses pre-trained classes)")
            except Exception as e:
                print(f"✅ Model loaded successfully!")
                print(f"   ⚠️  Could not set custom classes: {e}")
        else:
            # Custom model - use its pre-trained classes without restriction
            print(f"✅ Model loaded successfully!")
            print(f"   ℹ️  Using model's pre-trained classes (no custom class restriction)")
        
    def get_images_from_directory(self, directory):
        """Get all image files from a directory and its subdirectories."""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        
        # Check for images in the main directory
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory, ext)))
        
        # Also check subdirectories (train/images, test/images, valid/images)
        for subdir in ['train/images', 'test/images', 'valid/images', 'images']:
            subdir_path = os.path.join(directory, subdir)
            if os.path.exists(subdir_path):
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(subdir_path, ext)))
        
        return sorted(list(set(image_files)))  # Remove duplicates
    
    def evaluate_dataset(self, dataset_dir, expected_class, conf_threshold=0.25):
        """
        Evaluate model on a single test dataset.
        
        Args:
            dataset_dir: Path to test dataset directory
            expected_class: Expected class name for all objects
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary with evaluation metrics
        """
        dataset_name = os.path.basename(dataset_dir)
        print(f"\n{'─'*70}")
        print(f"📊 Evaluating: {dataset_name}")
        print(f"   Expected class: '{expected_class}'")
        print(f"   Directory: {dataset_dir}")
        print(f"{'─'*70}")
        
        if not os.path.exists(dataset_dir):
            print(f"⚠️  Directory not found: {dataset_dir}")
            return None
        
        # Get all images
        image_files = self.get_images_from_directory(dataset_dir)
        
        if not image_files:
            print(f"⚠️  No images found in {dataset_dir}")
            return None
        
        print(f"Found {len(image_files)} images to test")
        
        # Track metrics
        total_images = len(image_files)
        images_with_detections = 0
        images_with_correct_detections = 0
        total_detections = 0
        correct_detections = 0
        false_positives = 0
        
        # Confusion matrix elements
        true_positives = 0   # Correct class detected
        false_negatives = 0  # Expected class not detected
        
        # Process each image
        for idx, img_path in enumerate(image_files, 1):
            img_name = os.path.basename(img_path)
            
            # Run detection
            results = self.model.predict(
                source=img_path,
                conf=conf_threshold,
                verbose=False
            )
            
            result = results[0]
            detections = result.boxes
            
            # Track if this image has any detections
            has_detection = len(detections) > 0
            has_correct_detection = False
            
            if has_detection:
                images_with_detections += 1
                
                # Check each detection
                for box in detections:
                    class_id = int(box.cls[0])
                    
                    # Handle potential missing class names
                    if class_id not in result.names:
                        # Skip detections with unknown class IDs
                        continue
                    
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    
                    total_detections += 1
                    
                    # Check if detected class matches expected class (with synonym support)
                    is_correct = self._is_class_match(class_name, expected_class)
                    
                    if is_correct:
                        correct_detections += 1
                        has_correct_detection = True
                        true_positives += 1
                        
                        if idx <= 5:  # Show first 5 for verification
                            print(f"  [{idx:3d}] {img_name:40s} ✓ {class_name} ({confidence:.2f})")
                    else:
                        false_positives += 1
                        
                        if idx <= 5:
                            print(f"  [{idx:3d}] {img_name:40s} ✗ {class_name} (expected {expected_class})")
                
                if has_correct_detection:
                    images_with_correct_detections += 1
            else:
                # No detection = false negative
                false_negatives += 1
                if idx <= 5:
                    print(f"  [{idx:3d}] {img_name:40s} ✗ No detection")
        
        # Calculate metrics
        # Accuracy: percentage of images with correct detections
        accuracy = (images_with_correct_detections / total_images * 100) if total_images > 0 else 0
        
        # Detection rate: percentage of images with any detection
        detection_rate = (images_with_detections / total_images * 100) if total_images > 0 else 0
        
        # Precision: TP / (TP + FP)
        precision = (true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0
        
        # Recall: TP / (TP + FN)
        # For this task, we expect at least one object per image
        recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0
        
        # F1-score: 2 * (Precision * Recall) / (Precision + Recall)
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        # Store results
        metrics = {
            'dataset_name': dataset_name,
            'expected_class': expected_class,
            'total_images': total_images,
            'images_with_detections': images_with_detections,
            'images_with_correct_detections': images_with_correct_detections,
            'total_detections': total_detections,
            'correct_detections': correct_detections,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_positives': true_positives,
            'accuracy': accuracy,
            'detection_rate': detection_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
        # Print summary
        print(f"\n📈 Results for {dataset_name}:")
        print(f"   Total images: {total_images}")
        print(f"   Images with detections: {images_with_detections} ({detection_rate:.1f}%)")
        print(f"   Images with CORRECT detections: {images_with_correct_detections}")
        print(f"   Total detections: {total_detections}")
        print(f"   Correct detections: {correct_detections}")
        print(f"   False positives: {false_positives}")
        print(f"   False negatives: {false_negatives}")
        print(f"\n   ┌─ Performance Metrics ─────────────────┐")
        print(f"   │ Accuracy:   {accuracy:6.2f}%                     │")
        print(f"   │ Precision:  {precision:6.2f}                     │")
        print(f"   │ Recall:     {recall:6.2f}                     │")
        print(f"   │ F1-Score:   {f1_score:6.2f}                     │")
        print(f"   └───────────────────────────────────────┘")
        
        return metrics
    
    def evaluate_all_datasets(self, datasets_config):
        """
        Evaluate model on all test datasets.
        
        Args:
            datasets_config: List of dictionaries with 'path' and 'expected_class'
        """
        print(f"\n{'='*70}")
        print(f"🔍 EVALUATING MODEL: {self.model_name}")
        print(f"{'='*70}")
        
        all_accuracies = []
        all_f1_scores = []
        all_precisions = []
        all_recalls = []
        
        for dataset_config in datasets_config:
            dataset_dir = dataset_config['path']
            expected_class = dataset_config['expected_class']
            
            metrics = self.evaluate_dataset(dataset_dir, expected_class)
            
            if metrics:
                self.results['datasets'][dataset_config['name']] = metrics
                all_accuracies.append(metrics['accuracy'])
                all_f1_scores.append(metrics['f1_score'])
                all_precisions.append(metrics['precision'])
                all_recalls.append(metrics['recall'])
        
        # Calculate overall averages
        if all_accuracies:
            self.results['overall'] = {
                'average_accuracy': np.mean(all_accuracies),
                'average_precision': np.mean(all_precisions),
                'average_recall': np.mean(all_recalls),
                'average_f1_score': np.mean(all_f1_scores),
                'std_accuracy': np.std(all_accuracies),
                'std_f1_score': np.std(all_f1_scores)
            }
            
            print(f"\n{'='*70}")
            print(f"📊 OVERALL PERFORMANCE: {self.model_name}")
            print(f"{'='*70}")
            print(f"Average Accuracy:  {self.results['overall']['average_accuracy']:.2f}% (±{self.results['overall']['std_accuracy']:.2f}%)")
            print(f"Average Precision: {self.results['overall']['average_precision']:.4f}")
            print(f"Average Recall:    {self.results['overall']['average_recall']:.4f}")
            print(f"Average F1-Score:  {self.results['overall']['average_f1_score']:.4f} (±{self.results['overall']['std_f1_score']:.4f})")
            print(f"{'='*70}\n")
        
        return self.results


def compare_models(custom_results, standard_results):
    """
    Compare performance of two models.
    
    Args:
        custom_results: Results dictionary from custom model
        standard_results: Results dictionary from standard model
    """
    print(f"\n{'='*70}")
    print(f"⚖️  MODEL COMPARISON")
    print(f"{'='*70}\n")
    
    # Overall comparison
    print(f"{'Metric':<20} {'Custom YOLOv8x-world':<25} {'Standard YOLOv8x':<25} {'Difference':<15}")
    print(f"{'─'*20} {'─'*25} {'─'*25} {'─'*15}")
    
    custom_overall = custom_results.get('overall', {})
    standard_overall = standard_results.get('overall', {})
    
    metrics = [
        ('Average Accuracy', 'average_accuracy', '%'),
        ('Average Precision', 'average_precision', ''),
        ('Average Recall', 'average_recall', ''),
        ('Average F1-Score', 'average_f1_score', '')
    ]
    
    for metric_name, metric_key, unit in metrics:
        custom_val = custom_overall.get(metric_key, 0)
        standard_val = standard_overall.get(metric_key, 0)
        diff = custom_val - standard_val
        
        if unit == '%':
            custom_str = f"{custom_val:.2f}%"
            standard_str = f"{standard_val:.2f}%"
            diff_str = f"{diff:+.2f}%"
        else:
            custom_str = f"{custom_val:.4f}"
            standard_str = f"{standard_val:.4f}"
            diff_str = f"{diff:+.4f}"
        
        # Add color indicators
        if diff > 0:
            indicator = "📈 "
        elif diff < 0:
            indicator = "📉 "
        else:
            indicator = "➡️  "
        
        print(f"{metric_name:<20} {custom_str:<25} {standard_str:<25} {indicator}{diff_str:<15}")
    
    # Per-dataset comparison
    print(f"\n{'='*70}")
    print(f"📋 PER-DATASET COMPARISON")
    print(f"{'='*70}\n")
    
    custom_datasets = custom_results.get('datasets', {})
    standard_datasets = standard_results.get('datasets', {})
    
    for dataset_name in custom_datasets.keys():
        if dataset_name not in standard_datasets:
            continue
        
        custom_data = custom_datasets[dataset_name]
        standard_data = standard_datasets[dataset_name]
        
        print(f"Dataset: {dataset_name}")
        print(f"Expected class: {custom_data['expected_class']}")
        print(f"{'─'*70}")
        
        print(f"  {'Metric':<18} {'Custom':<15} {'Standard':<15} {'Difference':<15}")
        print(f"  {'─'*18} {'─'*15} {'─'*15} {'─'*15}")
        
        comparison_metrics = [
            ('Accuracy', 'accuracy', '%'),
            ('Precision', 'precision', ''),
            ('Recall', 'recall', ''),
            ('F1-Score', 'f1_score', '')
        ]
        
        for metric_name, metric_key, unit in comparison_metrics:
            custom_val = custom_data.get(metric_key, 0)
            standard_val = standard_data.get(metric_key, 0)
            diff = custom_val - standard_val
            
            if unit == '%':
                custom_str = f"{custom_val:.2f}%"
                standard_str = f"{standard_val:.2f}%"
                diff_str = f"{diff:+.2f}%"
            else:
                custom_str = f"{custom_val:.4f}"
                standard_str = f"{standard_val:.4f}"
                diff_str = f"{diff:+.4f}"
            
            print(f"  {metric_name:<18} {custom_str:<15} {standard_str:<15} {diff_str:<15}")
        
        print()
    
    # Summary verdict
    print(f"{'='*70}")
    print(f"🏆 VERDICT")
    print(f"{'='*70}")
    
    custom_f1 = custom_overall.get('average_f1_score', 0)
    standard_f1 = standard_overall.get('average_f1_score', 0)
    
    if custom_f1 > standard_f1:
        winner = "Custom YOLOv8x-world"
        improvement = ((custom_f1 - standard_f1) / standard_f1 * 100) if standard_f1 > 0 else 0
        print(f"✨ {winner} performs better!")
        print(f"   F1-Score improvement: {improvement:.2f}%")
    elif standard_f1 > custom_f1:
        winner = "Standard YOLOv8x"
        improvement = ((standard_f1 - custom_f1) / custom_f1 * 100) if custom_f1 > 0 else 0
        print(f"✨ {winner} performs better!")
        print(f"   F1-Score improvement: {improvement:.2f}%")
    else:
        print(f"⚖️  Both models perform equally well!")
    
    print(f"{'='*70}\n")


def save_results_to_json(custom_results, standard_results, output_file="model_comparison_results.json"):
    """Save evaluation results to JSON file."""
    results = {
        'custom_model': custom_results,
        'standard_model': standard_results,
        'comparison_summary': {
            'custom_f1': custom_results.get('overall', {}).get('average_f1_score', 0),
            'standard_f1': standard_results.get('overall', {}).get('average_f1_score', 0),
            'difference': custom_results.get('overall', {}).get('average_f1_score', 0) - 
                         standard_results.get('overall', {}).get('average_f1_score', 0)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")


def main():
    """Main evaluation function."""
    
    # Define test datasets and their expected classes
    datasets_config = [
        {
            'name': 'outlet_test_far',
            'path': 'outlet_test_far/outlet.v1i.yolov8',
            'expected_class': 'outlet'
        },
        {
            'name': 'socket_test_close',
            'path': 'socket_test_close/Ai project.v1i.yolov8',
            'expected_class': 'outlet'
        },
        {
            'name': 'fluor_lamp_test',
            'path': 'fluor_lamp_test/Fluorescent light detection.v1-original-with-augmentation.yolov8',
            'expected_class': 'fluorescent tube'
        },
        {
            'name': 'fire_extinguisher_test',
            'path': 'fire_extinguisher_test/Fire Extinguisher Finder.v1i.yolov8',
            'expected_class': 'fire extinguisher'
        }
    ]
    
    # Model paths
    custom_model_path = "custom_yolov8x.pt"
    standard_model_path = "yolov8x-world.pt"  # Will download if not present
    
    print(f"\n{'='*70}")
    print(f"🎯 COMPREHENSIVE MODEL EVALUATION AND COMPARISON")
    print(f"{'='*70}")
    print(f"\nTest Datasets:")
    for config in datasets_config:
        print(f"  • {config['name']:<25} → Expected: '{config['expected_class']}'")
    print(f"\nModels to Compare:")
    print(f"  1. Custom YOLOv8x-world: {custom_model_path}")
    print(f"  2. Standard YOLOv8x:     {standard_model_path}")
    print(f"{'='*70}\n")
    
    # Evaluate Custom Model
    print(f"\n{'#'*70}")
    print(f"# STEP 1: EVALUATE CUSTOM YOLOV8X-WORLD MODEL")
    print(f"{'#'*70}")
    
    custom_evaluator = ModelEvaluator(custom_model_path, "Custom YOLOv8x-world", use_custom_classes=False)
    custom_evaluator.load_model()
    custom_results = custom_evaluator.evaluate_all_datasets(datasets_config)
    
    # Evaluate Standard Model
    print(f"\n{'#'*70}")
    print(f"# STEP 2: EVALUATE STANDARD YOLOV8X MODEL")
    print(f"{'#'*70}")
    
    standard_evaluator = ModelEvaluator(standard_model_path, "Standard YOLOv8x-world", use_custom_classes=True)
    standard_evaluator.load_model()
    standard_results = standard_evaluator.evaluate_all_datasets(datasets_config)
    
    # Compare Models
    print(f"\n{'#'*70}")
    print(f"# STEP 3: COMPARE MODELS")
    print(f"{'#'*70}")
    
    compare_models(custom_results, standard_results)
    
    # Save results
    save_results_to_json(custom_results, standard_results)
    
    print(f"\n{'='*70}")
    print(f"✅ EVALUATION COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
