"""
MRI Registration Quality Validator

Purpose:
    Evaluate and validate the quality of registered MRI images
    Generate detailed quality assessment reports for patients and hospitals
    Calculate similarity metrics (NCC, Dice, etc.)
    Identify and flag low-quality registrations

Features:
    - Load PNG-format registered MRI images
    - Calculate multiple quality metrics
    - Generate patient-level detailed reports
    - Generate hospital-level aggregated reports
    - Automatic quality ranking and alerts

Output:
    - Patient reports: [hospital]_[patient_id]_registration_quality.png
    - Summary report: registration_quality_summary.png
    - JSON metadata with registration information

Usage:
    validator = RegistrationQualityValidator(data_root, output_dir)
    files, results = validator.validate_all_patients()

Author: MRI Processing Team
Version: 1.0
License: MIT
"""

import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import json
from tqdm import tqdm
import seaborn as sns
from scipy import ndimage
import warnings

warnings.filterwarnings('ignore')

# Set matplotlib backend and font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class RegistrationQualityValidator:
    """
    Validator for assessing MRI registration quality.
    
    This class provides functionality to load registered MRI images in PNG format,
    compute quality metrics, and generate comprehensive visualization reports.
    """
    
    def __init__(self, data_root, output_dir):
        """
        Initialize the validation engine.
        
        Args:
            data_root (Path or str): Root directory containing registered data
                Structure: data_root/[hospital]/[patient_id]/[modality]/ori|mask
            output_dir (Path or str): Directory for output reports
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Modality color mapping for visualization
        self.modality_colors = {
            'T1': '#FF6B6B',      # Red
            'T2': '#4ECDC4',      # Cyan
            'T1C': '#45B7D1'      # Blue
        }
        
        # Quality level thresholds
        self.quality_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.2
        }
        
        print(f"[INFO] Validator initialized")
        print(f"       Data root: {self.data_root}")
        print(f"       Output directory: {self.output_dir}")
    
    # ==================== Data Loading Module ====================
    
    def load_image_data(self, modality_dir):
        """
        Load image data from a modality directory.
        
        Args:
            modality_dir (Path): Directory containing ori/ and mask/ subdirectories
            
        Returns:
            dict: Dictionary containing:
                - ori_images: (N, H, W) numpy array of original images or None
                - mask_images: (M, H, W) numpy array of masks or None
                - slice_indices: List of slice indices
                - registration_info: Dict with registration metadata
        """
        ori_dir = modality_dir / 'ori'
        mask_dir = modality_dir / 'mask'
        info_file = modality_dir / 'registration_info.json'
        
        # Load registration metadata
        registration_info = {}
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    registration_info = json.load(f)
            except Exception as e:
                pass
        
        # Load original images
        ori_images = []
        slice_indices = []
        
        if ori_dir.exists():
            ori_files = sorted(list(ori_dir.glob('ori_*.png')))
            for file in ori_files:
                try:
                    img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        ori_images.append(img)
                        slice_idx = int(file.stem.split('_')[-1])
                        slice_indices.append(slice_idx)
                except Exception:
                    continue
        
        # Load mask images
        mask_images = []
        if mask_dir.exists():
            mask_files = sorted(list(mask_dir.glob('mask_*.png')))
            for file in mask_files:
                try:
                    img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        mask_images.append(img)
                except Exception:
                    continue
        
        return {
            'ori_images': np.array(ori_images) if ori_images else None,
            'mask_images': np.array(mask_images) if mask_images else None,
            'slice_indices': slice_indices,
            'registration_info': registration_info
        }
    
    # ==================== Similarity Calculation Module ====================
    
    def calculate_image_similarity(self, img1, img2):
        """
        Calculate Normalized Cross Correlation (NCC) between two images.
        
        NCC = Cov(X,Y) / (Std(X) * Std(Y))
        
        Args:
            img1, img2 (ndarray): Input images in uint8 format
            
        Returns:
            float: NCC similarity score in range [0, 1]
        """
        if img1 is None or img2 is None:
            return 0.0
        
        try:
            # Ensure same size
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Normalize to [0, 1]
            img1_norm = img1.astype(np.float32) / 255.0
            img2_norm = img2.astype(np.float32) / 255.0
            
            # Compute mean and std
            mean1, mean2 = np.mean(img1_norm), np.mean(img2_norm)
            std1, std2 = np.std(img1_norm), np.std(img2_norm)
            
            if std1 > 1e-6 and std2 > 1e-6:
                ncc = np.mean((img1_norm - mean1) * (img2_norm - mean2)) / (std1 * std2)
                return max(0.0, min(1.0, ncc))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def calculate_mask_overlap(self, mask1, mask2):
        """
        Calculate Dice coefficient between two masks.
        
        Dice = 2*|A intersect B| / (|A| + |B|)
        
        Args:
            mask1, mask2 (ndarray): Mask images in uint8 format
            
        Returns:
            float: Dice coefficient in range [0, 1]
        """
        if mask1 is None or mask2 is None:
            return 0.0
        
        try:
            # Ensure same size
            if mask1.shape != mask2.shape:
                mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))
            
            # Binarize with threshold 127
            mask1_bin = (mask1 > 127).astype(np.uint8)
            mask2_bin = (mask2 > 127).astype(np.uint8)
            
            # Compute Dice
            intersection = np.sum(mask1_bin * mask2_bin)
            union = np.sum(mask1_bin) + np.sum(mask2_bin)
            
            if union > 0:
                return 2.0 * intersection / union
            
            return 0.0
            
        except Exception:
            return 0.0
    
    # ==================== Data Analysis Module ====================
    
    def find_max_mask_slice(self, mask_images):
        """
        Find the slice with maximum mask area.
        
        Args:
            mask_images (ndarray): Array of mask images or None
            
        Returns:
            tuple: (slice_index, max_area) or (-1, 0) if no masks
        """
        if mask_images is None or len(mask_images) == 0:
            return -1, 0
        
        max_area = 0
        max_slice = -1
        
        for i, mask in enumerate(mask_images):
            mask_bin = (mask > 127).astype(np.uint8)
            area = np.sum(mask_bin)
            if area > max_area:
                max_area = area
                max_slice = i
        
        return max_slice, max_area
    
    def get_quality_color_and_label(self, score):
        """
        Get color and label for quality score visualization.
        
        Args:
            score (float): Quality score in [0, 1]
            
        Returns:
            tuple: (color_hex, label_text)
        """
        if score >= self.quality_thresholds['excellent']:
            return '#2ECC71', 'Excellent'
        elif score >= self.quality_thresholds['good']:
            return '#F39C12', 'Good'
        elif score >= self.quality_thresholds['fair']:
            return '#E74C3C', 'Fair'
        else:
            return '#95A5A6', 'Poor'
    
    # ==================== Image Processing Module ====================
    
    def create_overlay_image(self, img1, img2, alpha=0.5):
        """
        Create overlay visualization of two grayscale images.
        Second image is displayed in green channel.
        
        Args:
            img1, img2 (ndarray): Input grayscale images
            alpha (float): Blending factor for img2 (0-1)
            
        Returns:
            ndarray: RGB overlay image
        """
        if img1 is None or img2 is None:
            return img1 if img1 is not None else img2
        
        try:
            # Ensure same size
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Convert to RGB
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
            
            # Keep only green channel for img2
            img2_rgb[:, :, 0] = 0  # Remove red
            img2_rgb[:, :, 2] = 0  # Remove blue
            
            # Blend
            overlay = cv2.addWeighted(img1_rgb, 1-alpha, img2_rgb, alpha, 0)
            return overlay
            
        except Exception:
            return img1
    
    # ==================== Visualization Module ====================
    
    def visualize_patient_quality(self, hospital_id, patient_id, patient_data):
        """
        Generate detailed quality assessment visualization for a patient.
        
        Creates a comprehensive multi-panel figure showing:
        - Patient information and statistics
        - Original images (middle slice) with quality color-coded borders
        - Maximum mask slices
        - Quality metrics and comparisons
        
        Args:
            hospital_id (str): Hospital identifier
            patient_id (str): Patient identifier
            patient_data (dict): Patient data dictionary {modality: data}
            
        Returns:
            str: Path to generated figure
        """
        print(f"  Generating visualization: {hospital_id}-{patient_id}...")
        
        # Create figure
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 6, figure=fig,
                     height_ratios=[0.8, 1.5, 1.5, 1],
                     width_ratios=[1, 1, 1, 1, 1, 1])
        
        # Main title
        fig.suptitle(f'Registration Quality Report - Hospital: {hospital_id} - Patient: {patient_id}',
                    fontsize=20, fontweight='bold', y=0.95)
        
        modalities = list(patient_data.keys())
        n_modalities = len(modalities)
        
        # === Row 1: Summary Information ===
        ax_info = fig.add_subplot(gs[0, :])
        ax_info.axis('off')
        
        # Collect statistics
        total_slices = {}
        registration_qualities = {}
        mask_max_slices = {}
        
        for mod in modalities:
            data = patient_data[mod]
            total_slices[mod] = len(data['ori_images']) if data['ori_images'] is not None else 0
            
            # Get registration quality
            reg_info = data['registration_info']
            if 'avg_quality' in reg_info:
                registration_qualities[mod] = reg_info['avg_quality']
            elif 'quality_scores' in reg_info:
                registration_qualities[mod] = np.mean(reg_info['quality_scores'])
            else:
                registration_qualities[mod] = 0.0
            
            # Find max mask slice
            max_slice, _ = self.find_max_mask_slice(data['mask_images'])
            mask_max_slices[mod] = max_slice
        
        # Format info text
        info_text = f"Available modalities: {', '.join(modalities)} | "
        info_text += f"Total slices: {', '.join([f'{mod}({total_slices[mod]})' for mod in modalities])}\n"
        info_text += f"Registration quality: {', '.join([f'{mod}({registration_qualities[mod]:.3f})' for mod in modalities])}\n"
        info_text += f"Max mask slices: {', '.join([f'{mod}({mask_max_slices[mod]})' for mod in modalities])}"
        
        ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        
        # === Row 2: Original Images (Middle Slice) ===
        min_slice_count = min([len(data['ori_images'])
                              for data in patient_data.values()
                              if data['ori_images'] is not None])
        middle_slice_idx = min_slice_count // 2 if min_slice_count > 0 else 0
        
        for i, mod in enumerate(modalities):
            ax = fig.add_subplot(gs[1, i])
            data = patient_data[mod]
            
            if (data['ori_images'] is not None and
                middle_slice_idx < len(data['ori_images'])):
                
                img = data['ori_images'][middle_slice_idx]
                ax.imshow(img, cmap='gray')
                
                # Color border based on quality
                quality_score = registration_qualities[mod]
                color, label = self.get_quality_color_and_label(quality_score)
                
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(4)
                
                ax.set_title(f'{mod} - Middle Slice\nQuality: {quality_score:.3f} ({label})',
                           color=color, fontweight='bold', fontsize=12)
            else:
                ax.text(0.5, 0.5, f'{mod}\nNo Data', ha='center', va='center', fontsize=12)
                ax.set_facecolor('lightgray')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Overlay images (if multiple modalities)
        if n_modalities >= 2:
            for i in range(n_modalities, min(6, n_modalities + 2)):
                ax = fig.add_subplot(gs[1, i])
                
                if i == n_modalities and n_modalities >= 2:
                    mod1, mod2 = modalities[0], modalities[1]
                    data1, data2 = patient_data[mod1], patient_data[mod2]
                    
                    if (data1['ori_images'] is not None and
                        data2['ori_images'] is not None and
                        middle_slice_idx < len(data1['ori_images']) and
                        middle_slice_idx < len(data2['ori_images'])):
                        
                        overlay = self.create_overlay_image(
                            data1['ori_images'][middle_slice_idx],
                            data2['ori_images'][middle_slice_idx]
                        )
                        ax.imshow(overlay)
                        ax.set_title(f'{mod1} + {mod2} Overlay', fontweight='bold')
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        # === Row 3: Maximum Mask Slices ===
        for i, mod in enumerate(modalities):
            ax = fig.add_subplot(gs[2, i])
            data = patient_data[mod]
            max_slice_idx = mask_max_slices[mod]
            
            if (data['ori_images'] is not None and
                data['mask_images'] is not None and
                max_slice_idx >= 0 and max_slice_idx < len(data['ori_images'])):
                
                ori_img = data['ori_images'][max_slice_idx]
                mask_img = data['mask_images'][max_slice_idx]
                
                # Create RGB image with mask overlay
                colored_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2RGB)
                mask_colored = np.zeros_like(colored_img)
                mask_colored[:, :, 0] = mask_img  # Red channel for mask
                
                # Blend
                alpha = 0.3
                result = cv2.addWeighted(colored_img, 1-alpha, mask_colored, alpha, 0)
                ax.imshow(result)
                
                ax.set_title(f'{mod} - Max Mask Slice\nIndex: {max_slice_idx}',
                           fontweight='bold', fontsize=12)
            else:
                ax.text(0.5, 0.5, f'{mod}\nNo Mask Data', ha='center', va='center', fontsize=12)
                ax.set_facecolor('lightgray')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # === Row 4: Quality Analysis Charts ===
        
        # Quality scores bar chart
        ax_quality = fig.add_subplot(gs[3, :2])
        modality_names = list(registration_qualities.keys())
        quality_scores = list(registration_qualities.values())
        colors = [self.get_quality_color_and_label(score)[0] for score in quality_scores]
        
        bars = ax_quality.bar(modality_names, quality_scores, color=colors, alpha=0.7)
        ax_quality.set_ylabel('Quality Score')
        ax_quality.set_title('Registration Quality by Modality')
        ax_quality.set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars, quality_scores):
            height = bar.get_height()
            ax_quality.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add threshold lines
        for threshold_name, threshold_value in self.quality_thresholds.items():
            ax_quality.axhline(y=threshold_value, color='gray', linestyle='--', alpha=0.5)
        
        # Slice count comparison
        ax_slices = fig.add_subplot(gs[3, 2:4])
        slice_counts = list(total_slices.values())
        bars = ax_slices.bar(modality_names, slice_counts,
                            color=[self.modality_colors.get(mod, '#808080') for mod in modality_names],
                            alpha=0.7)
        ax_slices.set_ylabel('Slice Count')
        ax_slices.set_title('Slice Count by Modality')
        
        for bar, count in zip(bars, slice_counts):
            height = bar.get_height()
            ax_slices.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                          str(count), ha='center', va='bottom', fontweight='bold')
        
        # Max mask position comparison
        ax_mask_pos = fig.add_subplot(gs[3, 4:])
        mask_positions = [mask_max_slices[mod] for mod in modality_names]
        valid_positions = [pos for pos in mask_positions if pos >= 0]
        
        if valid_positions:
            bars = ax_mask_pos.bar(modality_names, mask_positions,
                                  color=[self.modality_colors.get(mod, '#808080') for mod in modality_names],
                                  alpha=0.7)
            ax_mask_pos.set_ylabel('Slice Index')
            ax_mask_pos.set_title('Max Mask Slice Position')
            
            for bar, pos in zip(bars, mask_positions):
                if pos >= 0:
                    height = bar.get_height()
                    ax_mask_pos.text(bar.get_x() + bar.get_width()/2., height + 1,
                                    str(pos), ha='center', va='bottom', fontweight='bold')
            
            # Check consistency
            if len(valid_positions) >= 2:
                max_diff = max(valid_positions) - min(valid_positions)
                consistency_text = f"Max difference: {max_diff} slices"
                
                if max_diff <= 3:
                    consistency_color = 'green'
                    consistency_text += " (Good)"
                elif max_diff <= 8:
                    consistency_color = 'orange'
                    consistency_text += " (Fair)"
                else:
                    consistency_color = 'red'
                    consistency_text += " (Poor)"
                
                ax_mask_pos.text(0.5, 0.95, consistency_text, transform=ax_mask_pos.transAxes,
                               ha='center', va='top', color=consistency_color, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax_mask_pos.text(0.5, 0.5, 'No Valid Mask Data', ha='center', va='center', fontsize=12)
            ax_mask_pos.set_facecolor('lightgray')
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_dir / f'{hospital_id}_{patient_id}_registration_quality.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_file)
    
    # ==================== Aggregation and Reporting Module ====================
    
    def calculate_cross_modality_similarity(self, patient_data):
        """
        Calculate cross-modality similarity metrics.
        
        Args:
            patient_data (dict): Patient data dictionary
            
        Returns:
            dict: Dictionary of modality pair similarities
        """
        modalities = list(patient_data.keys())
        similarities = {}
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities[i+1:], i+1):
                data1 = patient_data[mod1]
                data2 = patient_data[mod2]
                
                if (data1['ori_images'] is not None and data2['ori_images'] is not None):
                    # Compare common slices
                    min_slices = min(len(data1['ori_images']), len(data2['ori_images']))
                    if min_slices > 0:
                        start_idx = max(0, min_slices // 4)
                        end_idx = min(min_slices, 3 * min_slices // 4)
                        
                        sim_scores = []
                        for idx in range(start_idx, end_idx, max(1, (end_idx - start_idx) // 5)):
                            sim = self.calculate_image_similarity(
                                data1['ori_images'][idx],
                                data2['ori_images'][idx]
                            )
                            sim_scores.append(sim)
                        
                        if sim_scores:
                            similarities[f'{mod1}-{mod2}'] = np.mean(sim_scores)
        
        return similarities
    
    def generate_summary_report(self, all_results):
        """
        Generate comprehensive summary report across all hospitals and patients.
        
        Creates visualizations showing:
        - Quality distribution by hospital
        - Overall quality histogram
        - Modality count distribution
        - Success rates
        - Quality vs modality count correlation
        - Overall statistics
        
        Args:
            all_results (dict): Dictionary of all validation results
            {hospital_id: {patient_id: patient_data}}
            
        Returns:
            str: Path to generated summary figure
        """
        print("Generating summary report...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Registration Quality Summary Report', fontsize=20, fontweight='bold')
        
        # Collect statistics
        hospitals = []
        avg_qualities = []
        modality_counts = []
        successful_registrations = []
        
        for hospital_id, patients in all_results.items():
            for patient_id, patient_data in patients.items():
                hospitals.append(hospital_id)
                
                # Compute average quality
                qualities = []
                for mod_data in patient_data.values():
                    reg_info = mod_data['registration_info']
                    if 'avg_quality' in reg_info:
                        qualities.append(reg_info['avg_quality'])
                    elif 'quality_scores' in reg_info:
                        qualities.append(np.mean(reg_info['quality_scores']))
                
                avg_qualities.append(np.mean(qualities) if qualities else 0)
                modality_counts.append(len(patient_data))
                successful_registrations.append(len([q for q in qualities if q > 0.3]))
        
        # 1. Quality distribution by hospital
        ax = axes[0, 0]
        unique_hospitals = list(set(hospitals))
        hospital_qualities = {h: [avg_qualities[i] for i, hosp in enumerate(hospitals) if hosp == h]
                            for h in unique_hospitals}
        
        bp = ax.boxplot([hospital_qualities[h] for h in unique_hospitals],
                       labels=unique_hospitals, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_title('Quality Distribution by Hospital')
        ax.set_ylabel('Quality Score')
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Quality histogram
        ax = axes[0, 1]
        ax.hist(avg_qualities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('Quality Score Distribution')
        ax.set_xlabel('Quality Score')
        ax.set_ylabel('Patient Count')
        
        # Add threshold lines
        for threshold_name, threshold_value in self.quality_thresholds.items():
            ax.axvline(x=threshold_value, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        # 3. Modality count statistics
        ax = axes[0, 2]
        modality_count_stats = {i: modality_counts.count(i) for i in set(modality_counts)}
        bars = ax.bar(modality_count_stats.keys(), modality_count_stats.values(),
                     color='lightgreen', alpha=0.7)
        ax.set_title('Modality Count Distribution')
        ax.set_xlabel('Number of Modalities')
        ax.set_ylabel('Patient Count')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   str(int(height)), ha='center', va='bottom', fontweight='bold')
        
        # 4. Success rates
        ax = axes[1, 0]
        success_rates = {}
        for hospital in unique_hospitals:
            hospital_indices = [i for i, h in enumerate(hospitals) if h == hospital]
            total_modalities = sum(modality_counts[i] for i in hospital_indices)
            successful_modalities = sum(successful_registrations[i] for i in hospital_indices)
            success_rates[hospital] = successful_modalities / total_modalities if total_modalities > 0 else 0
        
        bars = ax.bar(success_rates.keys(), success_rates.values(),
                     color='orange', alpha=0.7)
        ax.set_title('Success Rate by Hospital')
        ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1)
        
        for bar, rate in zip(bars, success_rates.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Quality vs modality count scatter
        ax = axes[1, 1]
        ax.scatter(modality_counts, avg_qualities, alpha=0.6, s=100, c='steelblue')
        ax.set_xlabel('Number of Modalities')
        ax.set_ylabel('Average Quality Score')
        ax.set_title('Quality vs Modality Count')
        ax.grid(True, alpha=0.3)
        
        # 6. Statistics summary
        ax = axes[1, 2]
        ax.axis('off')
        
        total_patients = len(avg_qualities)
        excellent_count = sum(1 for q in avg_qualities if q >= self.quality_thresholds['excellent'])
        good_count = sum(1 for q in avg_qualities if q >= self.quality_thresholds['good'])
        
        stats_text = f"""Overall Statistics:

Total Patients: {total_patients}
Mean Quality: {np.mean(avg_qualities):.3f}
Std Quality: {np.std(avg_qualities):.3f}

Quality Distribution:
Excellent (>= 0.8): {excellent_count} ({excellent_count/total_patients*100:.1f}%)
Good (>= 0.6): {good_count} ({good_count/total_patients*100:.1f}%)

Hospitals: {len(unique_hospitals)}
"""
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5",
               facecolor="lightgray", alpha=0.8), family='monospace')
        
        plt.tight_layout()
        
        # Save figure
        summary_file = self.output_dir / 'registration_quality_summary.png'
        plt.savefig(summary_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(summary_file)
    
    # ==================== Main Validation Pipeline ====================
    
    def validate_all_patients(self, hospital_ids=None):
        """
        Validate registration quality for all patients across hospitals.
        
        Args:
            hospital_ids (list, optional): List of hospital IDs to process.
                If None, processes all subdirectories in data_root.
            
        Returns:
            tuple: (generated_files_list, all_results_dict)
        """
        # Discover hospitals
        if hospital_ids is None:
            hospital_ids = [d.name for d in self.data_root.iterdir() if d.is_dir()]
        
        hospital_ids.sort()
        
        all_results = {}
        generated_files = []
        
        print("[INFO] Starting registration quality validation")
        print(f"       Data root: {self.data_root}")
        print(f"       Output directory: {self.output_dir}")
        
        total_patients = 0
        processed_patients = 0
        
        # Count total patients
        for hospital_id in hospital_ids:
            hospital_dir = self.data_root / hospital_id
            if hospital_dir.exists():
                patient_dirs = [d for d in hospital_dir.iterdir() if d.is_dir()]
                total_patients += len(patient_dirs)
        
        with tqdm(total=total_patients, desc="Validating registration quality") as pbar:
            for hospital_id in hospital_ids:
                hospital_dir = self.data_root / hospital_id
                if not hospital_dir.exists():
                    print(f"[WARNING] Skipping non-existent hospital: {hospital_id}")
                    continue
                
                print(f"\n[INFO] Processing hospital: {hospital_id}")
                hospital_results = {}
                
                patient_dirs = [d for d in hospital_dir.iterdir() if d.is_dir()]
                
                for patient_dir in patient_dirs:
                    patient_id = patient_dir.name
                    pbar.set_description(f"Processing {hospital_id}-{patient_id}")
                    
                    try:
                        # Load patient data
                        patient_data = {}
                        modalities = ['T1', 'T2', 'T1C']
                        
                        for modality in modalities:
                            modality_dir = patient_dir / modality
                            if modality_dir.exists():
                                modality_data = self.load_image_data(modality_dir)
                                if modality_data['ori_images'] is not None:
                                    patient_data[modality] = modality_data
                        
                        # Process patient with valid data
                        if len(patient_data) >= 1:
                            hospital_results[patient_id] = patient_data
                            
                            # Generate patient quality visualization
                            output_file = self.visualize_patient_quality(
                                hospital_id, patient_id, patient_data
                            )
                            generated_files.append(output_file)
                            processed_patients += 1
                            
                            print(f"  [OK] {patient_id}: {len(patient_data)} modalities")
                        else:
                            print(f"  [SKIP] {patient_id}: No valid data")
                    
                    except Exception as e:
                        print(f"  [ERROR] {patient_id}: {e}")
                    
                    pbar.update(1)
                
                if hospital_results:
                    all_results[hospital_id] = hospital_results
                    print(f"  Completed: {len(hospital_results)} patients")
        
        # Generate summary report
        if all_results:
            summary_file = self.generate_summary_report(all_results)
            generated_files.append(summary_file)
            print(f"\n[INFO] Summary report generated: {summary_file}")
        
        print(f"\n[INFO] Validation complete")
        print(f"       Processed: {processed_patients}/{total_patients} patients")
        print(f"       Generated: {len(generated_files)} files")
        print(f"       Output: {self.output_dir}")
        
        return generated_files, all_results


def main():
    """
    Main entry point for registration quality validation.
    
    Configurable parameters:
    - data_root: Root directory containing registered MRI data
    - output_dir: Output directory for quality reports
    """
    
    # Configuration
    data_root = Path("./data/registered")
    output_dir = Path("./quality_reports")
    
    # Create validator
    validator = RegistrationQualityValidator(data_root, output_dir)
    
    # Execute validation
    try:
        generated_files, results = validator.validate_all_patients()
        
        print(f"\n{'='*70}")
        print("REGISTRATION QUALITY VALIDATION SUMMARY")
        print(f"{'='*70}")
        
        # Compute overall statistics
        total_hospitals = len(results)
        total_patients = sum(len(patients) for patients in results.values())
        total_modalities = sum(len(patient_data)
                             for patients in results.values()
                             for patient_data in patients.values())
        
        print(f"Validation Results:")
        print(f"  Hospitals: {total_hospitals}")
        print(f"  Patients: {total_patients}")
        print(f"  Modalities: {total_modalities}")
        print(f"  Generated files: {len(generated_files)}")
        
        # Quality statistics
        all_qualities = []
        modality_stats = {}
        
        for hospital_data in results.values():
            for patient_data in hospital_data.values():
                for modality, mod_data in patient_data.items():
                    modality_stats[modality] = modality_stats.get(modality, 0) + 1
                    
                    reg_info = mod_data['registration_info']
                    if 'avg_quality' in reg_info:
                        all_qualities.append(reg_info['avg_quality'])
                    elif 'quality_scores' in reg_info:
                        all_qualities.append(np.mean(reg_info['quality_scores']))
        
        if all_qualities:
            print(f"\nQuality Statistics:")
            print(f"  Mean: {np.mean(all_qualities):.3f}")
            print(f"  Range: {np.min(all_qualities):.3f} - {np.max(all_qualities):.3f}")
            print(f"  Std: {np.std(all_qualities):.3f}")
            
            # Quality levels
            excellent = sum(1 for q in all_qualities if q >= 0.8)
            good = sum(1 for q in all_qualities if q >= 0.6)
            fair = sum(1 for q in all_qualities if q >= 0.4)
            poor = len(all_qualities) - fair
            
            print(f"\nQuality Levels:")
            print(f"  Excellent (>= 0.8): {excellent} ({excellent/len(all_qualities)*100:.1f}%)")
            print(f"  Good (>= 0.6): {good-excellent} ({(good-excellent)/len(all_qualities)*100:.1f}%)")
            print(f"  Fair (>= 0.4): {fair-good} ({(fair-good)/len(all_qualities)*100:.1f}%)")
            print(f"  Poor (< 0.4): {poor} ({poor/len(all_qualities)*100:.1f}%)")
        
        print(f"\nModality Distribution:")
        for modality, count in sorted(modality_stats.items()):
            print(f"  {modality}: {count}")
        
        print(f"\nOutput: {output_dir}")
        print(f"{'='*70}")
        print("VALIDATION COMPLETED")
        print(f"{'='*70}")
        
    except KeyboardInterrupt:
        print("\n[WARNING] Validation interrupted by user")
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
