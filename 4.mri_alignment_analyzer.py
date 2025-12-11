import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
import warnings

warnings.filterwarnings('ignore')


class AlignmentAnalyzer:
    """Analyze multi-modal MRI image alignment quality"""
    
    @staticmethod
    def calculate_mutual_information(img1, img2):
        """Calculate mutual information between two images
        
        Args:
            img1: First image array
            img2: Second image array
            
        Returns:
            Normalized mutual information score
        """
        # Flatten images and remove NaN values
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        
        # Remove NaN and Inf values
        valid_mask = np.isfinite(img1_flat) & np.isfinite(img2_flat)
        img1_valid = img1_flat[valid_mask]
        img2_valid = img2_flat[valid_mask]
        
        if len(img1_valid) == 0:
            return 0
        
        # Normalize to 0-255 range for histogram calculation
        img1_norm = ((img1_valid - img1_valid.min()) / 
                     (img1_valid.max() - img1_valid.min() + 1e-8) * 255).astype(int)
        img2_norm = ((img2_valid - img2_valid.min()) / 
                     (img2_valid.max() - img2_valid.min() + 1e-8) * 255).astype(int)
        
        return normalized_mutual_info_score(img1_norm, img2_norm)
    
    @staticmethod
    def calculate_dice_coefficient(mask1, mask2):
        """Calculate Dice coefficient between two masks
        
        Args:
            mask1: First binary mask
            mask2: Second binary mask
            
        Returns:
            Dice coefficient (0 to 1)
        """
        intersection = np.sum(mask1 * mask2)
        union = np.sum(mask1) + np.sum(mask2)
        if union == 0:
            return 1.0  # Both masks are empty, considered perfectly matched
        return 2.0 * intersection / union
    
    @staticmethod
    def check_spatial_alignment(nii1, nii2):
        """Check spatial alignment between two NIfTI files
        
        Args:
            nii1: First NIfTI image object
            nii2: Second NIfTI image object
            
        Returns:
            Dictionary with alignment information
        """
        # Check image shape
        shape_match = nii1.shape == nii2.shape
        
        # Check voxel size
        pixdim1 = nii1.header.get_zooms()
        pixdim2 = nii2.header.get_zooms()
        pixdim_match = np.allclose(pixdim1, pixdim2, rtol=1e-3)
        
        # Check affine matrix
        affine_match = np.allclose(nii1.affine, nii2.affine, rtol=1e-3)
        
        return {
            'shape_match': shape_match,
            'pixdim_match': pixdim_match,
            'affine_match': affine_match,
            'shape1': nii1.shape,
            'shape2': nii2.shape,
            'pixdim1': pixdim1,
            'pixdim2': pixdim2
        }
    
    @staticmethod
    def resize_to_match(img, target_shape):
        """Resize image to target shape
        
        Args:
            img: Image array
            target_shape: Target shape tuple
            
        Returns:
            Resized image array
        """
        if img.shape == target_shape:
            return img
        
        zoom_factors = [t/s for t, s in zip(target_shape, img.shape)]
        return zoom(img, zoom_factors, order=1)
    
    def analyze_patient_alignment(self, patient_path):
        """Analyze alignment quality for a single patient
        
        Args:
            patient_path: Path to patient directory
            
        Returns:
            Dictionary with alignment analysis results
        """
        modalities = ['T1', 'T2', 'T1C']
        available_modalities = []
        modality_data = {}
        
        # Check available modalities
        for modality in modalities:
            modality_path = os.path.join(patient_path, modality)
            if os.path.exists(modality_path):
                ori_path = os.path.join(modality_path, 'ori.nii.gz')
                mask_path = os.path.join(modality_path, 'mask.nii.gz')
                
                if os.path.exists(ori_path) and os.path.exists(mask_path):
                    try:
                        ori_nii = nib.load(ori_path)
                        mask_nii = nib.load(mask_path)
                        modality_data[modality] = {
                            'ori_nii': ori_nii,
                            'mask_nii': mask_nii,
                            'ori_data': ori_nii.get_fdata(),
                            'mask_data': mask_nii.get_fdata()
                        }
                        available_modalities.append(modality)
                    except Exception as e:
                        print(f"Error loading {modality}: {e}")
        
        if len(available_modalities) < 2:
            return {
                'patient_id': os.path.basename(patient_path),
                'available_modalities': available_modalities,
                'alignment_needed': False,
                'reason': 'Less than 2 modalities available'
            }
        
        # Analyze inter-modality alignment
        alignment_results = {}
        
        for i, mod1 in enumerate(available_modalities):
            for mod2 in available_modalities[i+1:]:
                pair_key = f"{mod1}_{mod2}"
                
                # Check spatial alignment
                spatial_info = self.check_spatial_alignment(
                    modality_data[mod1]['ori_nii'], 
                    modality_data[mod2]['ori_nii']
                )
                
                # Get image data
                img1 = modality_data[mod1]['ori_data']
                img2 = modality_data[mod2]['ori_data']
                mask1 = modality_data[mod1]['mask_data']
                mask2 = modality_data[mod2]['mask_data']
                
                # Resize to same size if needed
                if not spatial_info['shape_match']:
                    target_shape = tuple(min(s1, s2) for s1, s2 in zip(img1.shape, img2.shape))
                    img1_resized = self.resize_to_match(img1, target_shape)
                    img2_resized = self.resize_to_match(img2, target_shape)
                    mask1_resized = self.resize_to_match(mask1, target_shape)
                    mask2_resized = self.resize_to_match(mask2, target_shape)
                else:
                    img1_resized = img1
                    img2_resized = img2
                    mask1_resized = mask1
                    mask2_resized = mask2
                
                # Calculate mutual information
                mutual_info = self.calculate_mutual_information(img1_resized, img2_resized)
                
                # Calculate mask overlap (Dice coefficient)
                mask1_binary = (mask1_resized > 0).astype(int)
                mask2_binary = (mask2_resized > 0).astype(int)
                dice_coeff = self.calculate_dice_coefficient(mask1_binary, mask2_binary)
                
                # Determine if well aligned
                is_well_aligned = (
                    spatial_info['shape_match'] and 
                    spatial_info['pixdim_match'] and 
                    spatial_info['affine_match'] and
                    dice_coeff > 0.7 and
                    mutual_info > 0.1
                )
                
                alignment_results[pair_key] = {
                    'spatial_alignment': spatial_info,
                    'mutual_information': mutual_info,
                    'mask_dice_coefficient': dice_coeff,
                    'well_aligned': is_well_aligned
                }
        
        # Determine if registration is needed
        alignment_needed = False
        poorly_aligned_pairs = []
        
        for pair, result in alignment_results.items():
            if not result['well_aligned']:
                alignment_needed = True
                poorly_aligned_pairs.append(pair)
        
        return {
            'patient_id': os.path.basename(patient_path),
            'available_modalities': available_modalities,
            'alignment_results': alignment_results,
            'alignment_needed': alignment_needed,
            'poorly_aligned_pairs': poorly_aligned_pairs
        }
    
    def analyze_all_patients(self, base_path, datasets=None):
        """Analyze alignment for all patients in dataset
        
        Args:
            base_path: Base directory path
            datasets: List of dataset names (auto-detect if None)
            
        Returns:
            List of analysis results for all patients
        """
        if datasets is None:
            # Auto-detect datasets
            datasets = [d for d in os.listdir(base_path) 
                       if os.path.isdir(os.path.join(base_path, d))]
        
        all_results = []
        
        for dataset in datasets:
            dataset_path = os.path.join(base_path, dataset)
            if not os.path.exists(dataset_path):
                print(f"Dataset path not found: {dataset_path}")
                continue
            
            print(f"Processing dataset: {dataset}")
            
            # Get all patient folders
            patient_folders = [f for f in os.listdir(dataset_path) 
                              if os.path.isdir(os.path.join(dataset_path, f))]
            
            for patient_folder in patient_folders:
                patient_path = os.path.join(dataset_path, patient_folder)
                print(f"  Processing patient: {patient_folder}")
                
                result = self.analyze_patient_alignment(patient_path)
                result['dataset'] = dataset
                all_results.append(result)
        
        return all_results


def generate_summary_report(results):
    """Generate summary report of alignment analysis
    
    Args:
        results: List of analysis results
    """
    print("\n" + "="*80)
    print("Image Alignment Quality Analysis Report")
    print("="*80)
    
    total_patients = len(results)
    patients_needing_registration = sum(1 for r in results if r['alignment_needed'])
    
    print(f"Total patients: {total_patients}")
    print(f"Patients needing registration: {patients_needing_registration}")
    print(f"Percentage needing registration: {patients_needing_registration/total_patients*100:.1f}%")
    
    # Statistics by dataset
    print("\nStatistics by dataset:")
    dataset_stats = {}
    for result in results:
        dataset = result['dataset']
        if dataset not in dataset_stats:
            dataset_stats[dataset] = {'total': 0, 'need_registration': 0}
        dataset_stats[dataset]['total'] += 1
        if result['alignment_needed']:
            dataset_stats[dataset]['need_registration'] += 1
    
    for dataset, stats in sorted(dataset_stats.items()):
        ratio = stats['need_registration'] / stats['total'] * 100
        print(f"  {dataset}: {stats['need_registration']}/{stats['total']} ({ratio:.1f}%)")
    
    # Detailed results for patients needing registration
    print("\nPatients needing registration:")
    for result in results:
        if result['alignment_needed']:
            print(f"\n  Patient: {result['dataset']}/{result['patient_id']}")
            print(f"    Available modalities: {result['available_modalities']}")
            print(f"    Poorly aligned pairs: {result['poorly_aligned_pairs']}")
            
            if 'alignment_results' in result:
                for pair, align_info in result['alignment_results'].items():
                    if not align_info['well_aligned']:
                        spatial = align_info['spatial_alignment']
                        print(f"      {pair}:")
                        print(f"        Spatial alignment: "
                              f"shape={spatial['shape_match']}, "
                              f"spacing={spatial['pixdim_match']}, "
                              f"affine={spatial['affine_match']}")
                        print(f"        Mutual information: {align_info['mutual_information']:.3f}")
                        print(f"        Mask Dice coefficient: {align_info['mask_dice_coefficient']:.3f}")


def save_detailed_results(results, output_file):
    """Save detailed analysis results to CSV
    
    Args:
        results: List of analysis results
        output_file: Path to output CSV file
    """
    detailed_results = []
    for result in results:
        if 'alignment_results' in result:
            for pair, align_info in result['alignment_results'].items():
                detailed_results.append({
                    'dataset': result['dataset'],
                    'patient_id': result['patient_id'],
                    'modality_pair': pair,
                    'shape_match': align_info['spatial_alignment']['shape_match'],
                    'pixdim_match': align_info['spatial_alignment']['pixdim_match'],
                    'affine_match': align_info['spatial_alignment']['affine_match'],
                    'mutual_information': align_info['mutual_information'],
                    'mask_dice_coefficient': align_info['mask_dice_coefficient'],
                    'well_aligned': align_info['well_aligned']
                })
    
    if detailed_results:
        df = pd.DataFrame(detailed_results)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nDetailed results saved to: {output_file}")


def main():
    """Main entry point"""
    print("🔍 MRI Image Alignment Analyzer")
    print("="*80)
    
    # Configuration
    base_path = "./data/bias_corrected"
    output_file = "alignment_analysis_results.csv"
    
    print(f"📂 Input directory: {base_path}")
    
    # Analyze alignment
    print("\nAnalyzing image alignment quality...\n")
    analyzer = AlignmentAnalyzer()
    results = analyzer.analyze_all_patients(base_path)
    
    # Generate report
    generate_summary_report(results)
    
    # Save detailed results
    save_detailed_results(results, output_file)
    
    print("\n✨ Analysis completed!")


if __name__ == "__main__":
    main()
