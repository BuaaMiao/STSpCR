import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
import json
from tqdm import tqdm
import shutil
from datetime import datetime


class BiasFieldCorrector:
    """Correct intensity inhomogeneity (bias field) in MRI images using N4ITK"""
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file
        self.log_file = self.output_dir / "bias_correction_log.txt"
        self.processing_report = {}
    
    def log_message(self, message):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')
    
    def find_all_patients(self):
        """Find all patient directories in dataset"""
        all_patients = []
        
        for item in self.input_dir.iterdir():
            if item.is_dir():
                dataset_name = item.name
                for subitem in item.iterdir():
                    if subitem.is_dir():
                        modality_dirs = [d for d in subitem.iterdir() 
                                       if d.is_dir() and d.name in ['T1', 'T2', 'T1C']]
                        if modality_dirs:
                            patient_id = f"{dataset_name}_{subitem.name}"
                            all_patients.append({
                                'patient_id': patient_id,
                                'dataset': dataset_name,
                                'path': subitem
                            })
        
        return all_patients
    
    def bias_field_correction(self, image, mask=None):
        """Apply N4ITK bias field correction to image
        
        Args:
            image: SimpleITK image object
            mask: Optional mask image for correction region
            
        Returns:
            Tuple of (corrected_image, success_flag)
        """
        try:
            # Convert to float32
            input_image = sitk.Cast(image, sitk.sitkFloat32)
            
            # Create N4 bias field correction filter
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            
            # Set parameters
            corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
            corrector.SetBiasFieldFullWidthAtHalfMaximum(0.15)
            corrector.SetWienerFilterNoise(0.01)
            corrector.SetNumberOfHistogramBins(200)
            corrector.SetConvergenceThreshold(0.001)
            
            # Execute correction
            if mask is not None:
                mask_image = sitk.Cast(mask, sitk.sitkUInt8)
                corrected_image = corrector.Execute(input_image, mask_image)
            else:
                corrected_image = corrector.Execute(input_image)
            
            return corrected_image, True
            
        except Exception as e:
            print(f"Bias field correction failed: {str(e)}")
            return image, False
    
    def process_single_patient(self, patient_info):
        """Process bias field correction for a single patient"""
        patient_id = patient_info['patient_id']
        patient_path = patient_info['path']
        dataset = patient_info['dataset']
        
        try:
            self.log_message(f"Processing patient: {patient_id}")
            
            # Create output directory
            output_patient_dir = self.output_dir / dataset / patient_path.name
            output_patient_dir.mkdir(parents=True, exist_ok=True)
            
            processed_modalities = []
            failed_modalities = []
            
            # Process each modality
            for modality in ['T1', 'T2', 'T1C']:
                modality_path = patient_path / modality
                ori_file = modality_path / "ori.nii.gz"
                mask_file = modality_path / "mask.nii.gz"
                
                if ori_file.exists():
                    try:
                        # Read image
                        original_image = sitk.ReadImage(str(ori_file))
                        original_mask = None
                        if mask_file.exists():
                            original_mask = sitk.ReadImage(str(mask_file))
                        
                        # Execute bias field correction
                        corrected_image, success = self.bias_field_correction(
                            original_image, original_mask
                        )
                        
                        # Create modality output directory
                        modality_output_dir = output_patient_dir / modality
                        modality_output_dir.mkdir(exist_ok=True)
                        
                        if success:
                            # Save corrected image
                            output_ori_path = modality_output_dir / "ori.nii.gz"
                            sitk.WriteImage(corrected_image, str(output_ori_path))
                            
                            # Copy mask
                            if original_mask is not None:
                                output_mask_path = modality_output_dir / "mask.nii.gz"
                                sitk.WriteImage(original_mask, str(output_mask_path))
                            
                            processed_modalities.append(modality)
                            self.log_message(f"  ✅ {modality} bias field correction successful")
                        else:
                            # Copy original files if correction failed
                            output_ori_path = modality_output_dir / "ori.nii.gz"
                            shutil.copy2(ori_file, output_ori_path)
                            if mask_file.exists():
                                output_mask_path = modality_output_dir / "mask.nii.gz"
                                shutil.copy2(mask_file, output_mask_path)
                            
                            failed_modalities.append(modality)
                            self.log_message(f"  ❌ {modality} bias field correction failed, using original")
                    
                    except Exception as e:
                        self.log_message(f"  ❌ {modality} processing error: {str(e)}")
                        failed_modalities.append(modality)
            
            return {
                'patient_id': patient_id,
                'status': 'success' if processed_modalities else 'error',
                'processed_modalities': processed_modalities,
                'failed_modalities': failed_modalities
            }
                
        except Exception as e:
            self.log_message(f"❌ Patient {patient_id} processing failed: {str(e)}")
            return {
                'patient_id': patient_id,
                'status': 'error',
                'message': str(e)
            }
    
    def run_bias_correction(self):
        """Run batch bias field correction for all patients"""
        self.log_message("🔧 Starting batch bias field correction...")
        self.log_message(f"Input directory: {self.input_dir}")
        self.log_message(f"Output directory: {self.output_dir}")
        
        # Find all patients
        all_patients = self.find_all_patients()
        self.log_message(f"Found {len(all_patients)} patients")
        
        if not all_patients:
            self.log_message("❌ No patient data found")
            return
        
        # Process patients
        successful_count = 0
        failed_count = 0
        
        for patient_info in tqdm(all_patients, desc="Processing bias correction"):
            result = self.process_single_patient(patient_info)
            
            if result['status'] == 'success':
                successful_count += 1
            else:
                failed_count += 1
            
            self.processing_report[result['patient_id']] = result
        
        # Generate report
        self.generate_report()
        
        self.log_message(f"\n🎉 Bias field correction complete!")
        self.log_message(f"✅ Successfully processed: {successful_count} patients")
        self.log_message(f"❌ Processing failed: {failed_count} patients")
    
    def generate_report(self):
        """Generate processing report with visualizations"""
        # Save detailed report
        report_file = self.output_dir / f"bias_correction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.processing_report, f, indent=2, ensure_ascii=False)
        
        # Collect statistics
        total_patients = len(self.processing_report)
        successful_patients = sum(1 for r in self.processing_report.values() if r['status'] == 'success')
        
        # Create visualization report
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Success rate pie chart
        ax1.pie([successful_patients, total_patients - successful_patients], 
                labels=['Successful', 'Failed'], autopct='%1.1f%%', 
                colors=['lightgreen', 'lightcoral'])
        ax1.set_title(f'Bias Field Correction Results\n(Total: {total_patients} patients)', 
                     fontweight='bold')
        
        # Modality statistics bar chart
        modality_counts = {'T1': 0, 'T2': 0, 'T1C': 0}
        for result in self.processing_report.values():
            if result['status'] == 'success':
                for modality in result.get('processed_modalities', []):
                    modality_counts[modality] += 1
        
        ax2.bar(modality_counts.keys(), modality_counts.values(), 
                color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_title('Modality-wise Processing Count', fontweight='bold')
        ax2.set_ylabel('Number of Patients')
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.output_dir / f"bias_correction_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log_message(f"📊 Report saved: {report_file}")
        self.log_message(f"📊 Visualization saved: {viz_file}")


def main():
    """Main entry point"""
    print("🔧 MRI Bias Field Correction Tool")
    print("="*60)
    
    # Configuration
    input_dir = "./data/output"
    output_dir = "./data/bias_corrected"
    
    print(f"📂 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Create corrector and run
    corrector = BiasFieldCorrector(input_dir, output_dir)
    corrector.run_bias_correction()
    
    print("\n✨ Bias field correction completed!")
    print(f"Processed data saved to: {output_dir}")


if __name__ == "__main__":
    main()
