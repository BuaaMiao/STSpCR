import os
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import SimpleITK as sitk
from tqdm import tqdm
import logging
import shutil
import traceback
import numpy as np


def process_single_patient_worker(patient_info, base_data_dir, output_dir, target_spacing):
    """Process a single patient's MRI data"""
    from pathlib import Path
    import shutil
    
    patient_id = patient_info['patient_id']
    patient_path = patient_info['path']
    dataset = patient_info['dataset']
    
    base_data_dir = Path(base_data_dir)
    output_dir = Path(output_dir)
    
    # Temporary output directory
    temp_output_dir = output_dir / "temp" / dataset / patient_path.name
    final_output_dir = output_dir / dataset / patient_path.name
    
    try:
        print(f"\n{'='*60}")
        print(f"Processing patient: {patient_id}")
        print(f"{'='*60}")
        
        # Check available modalities
        modalities = {}
        for modality in ['T1', 'T2', 'T1C']:
            modality_path = patient_path / modality
            if modality_path.exists():
                ori_file = modality_path / "ori.nii.gz"
                mask_file = modality_path / "mask.nii.gz"
                
                if ori_file.exists():
                    modalities[modality] = {
                        'ori': ori_file,
                        'mask': mask_file if mask_file.exists() else None
                    }
        
        if not modalities:
            error_msg = f"No valid modalities found for patient {patient_id}"
            print(error_msg)
            return {'patient_id': patient_id, 'status': 'error', 'error': error_msg}
        
        print(f"Found modalities: {list(modalities.keys())}")
        
        # Create temporary output directory
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing results record
        processing_results = {
            'modalities_processed': [],
            'modalities_failed': [],
            'target_spacing': target_spacing
        }
        
        # Find reference image's spatial information
        reference_modality = 'T1' if 'T1' in modalities else list(modalities.keys())[0]
        print(f"\nUsing {reference_modality} as reference modality")
        
        # Read reference image
        reference_image = sitk.ReadImage(str(modalities[reference_modality]['ori']))
        reference_origin = reference_image.GetOrigin()
        reference_direction = reference_image.GetDirection()
        reference_dimension = reference_image.GetDimension()
        
        # Calculate resampled size based on target spacing
        original_spacing = reference_image.GetSpacing()
        original_size = reference_image.GetSize()
        reference_size = [
            int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
            for i in range(3)
        ]
        
        print(f"Reference image info:")
        print(f"  Original size: {original_size}")
        print(f"  Original spacing: {original_spacing}")
        print(f"  Dimension: {reference_dimension}")
        print(f"  Target size: {reference_size}")
        print(f"  Target spacing: {target_spacing}")
        print(f"  Origin: {reference_origin}")
        
        # Process each modality
        for modality, files in modalities.items():
            print(f"\nProcessing {modality} modality...")
            
            try:
                # Read original image
                ori_image = sitk.ReadImage(str(files['ori']))
                image_dimension = ori_image.GetDimension()
                image_size = ori_image.GetSize()
                image_spacing = ori_image.GetSpacing()
                
                print(f"  Original info: dimension={image_dimension}, size={image_size}, spacing={image_spacing}")
                
                # Create resampler with reference image's spatial info
                resampler = sitk.ResampleImageFilter()
                resampler.SetSize(reference_size)
                resampler.SetOutputSpacing(target_spacing)
                resampler.SetOutputOrigin(reference_origin)
                resampler.SetOutputDirection(reference_direction)
                resampler.SetInterpolator(sitk.sitkLinear)
                resampler.SetDefaultPixelValue(0)
                
                # Handle alignment for non-reference modalities
                if modality != reference_modality:
                    if image_dimension != reference_dimension:
                        print(f"  Warning: dimension mismatch, using identity transform")
                        identity = sitk.Transform(3, sitk.sitkIdentity)
                        resampler.SetTransform(identity)
                    else:
                        try:
                            print(f"  Performing geometric alignment...")
                            transform = sitk.CenteredTransformInitializer(
                                reference_image, 
                                ori_image,
                                sitk.Euler3DTransform(),
                                sitk.CenteredTransformInitializerFilter.GEOMETRY
                            )
                            resampler.SetTransform(transform)
                            print(f"  Geometric alignment successful")
                        except Exception as e:
                            print(f"  Geometric alignment failed: {str(e)}")
                            identity = sitk.Transform(3, sitk.sitkIdentity)
                            resampler.SetTransform(identity)
                
                # Execute resampling
                print(f"  Executing resampling...")
                processed_ori = resampler.Execute(ori_image)
                
                print(f"  {modality} processing completed:")
                print(f"    Output size: {processed_ori.GetSize()}")
                print(f"    Output spacing: {processed_ori.GetSpacing()}")
                print(f"    Output origin: {processed_ori.GetOrigin()}")
                
                # Validate data
                array = sitk.GetArrayFromImage(processed_ori)
                print(f"    Data range: [{np.min(array):.2f}, {np.max(array):.2f}]")
                print(f"    Non-zero voxels: {np.count_nonzero(array)}/{array.size}")
                
                # Process mask if exists
                processed_mask = None
                if files['mask'] is not None:
                    try:
                        mask_image = sitk.ReadImage(str(files['mask']))
                        
                        mask_resampler = sitk.ResampleImageFilter()
                        mask_resampler.SetSize(reference_size)
                        mask_resampler.SetOutputSpacing(target_spacing)
                        mask_resampler.SetOutputOrigin(reference_origin)
                        mask_resampler.SetOutputDirection(reference_direction)
                        mask_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                        mask_resampler.SetDefaultPixelValue(0)
                        
                        # Use same transform as image
                        if modality != reference_modality:
                            if image_dimension != reference_dimension:
                                mask_resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
                            else:
                                try:
                                    mask_resampler.SetTransform(transform)
                                except:
                                    mask_resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
                        
                        processed_mask = mask_resampler.Execute(mask_image)
                        print(f"    Mask processed successfully")
                        
                    except Exception as e:
                        print(f"    Mask processing failed: {str(e)}")
                
                # Create modality output directory
                modality_output_dir = temp_output_dir / modality
                modality_output_dir.mkdir(exist_ok=True)
                
                # Save processed image
                output_ori_path = modality_output_dir / "ori.nii.gz"
                sitk.WriteImage(processed_ori, str(output_ori_path))
                print(f"    Saved to: {output_ori_path}")
                
                # Check output file size
                file_size_mb = output_ori_path.stat().st_size / (1024*1024)
                print(f"    File size: {file_size_mb:.1f} MB")
                
                if processed_mask is not None:
                    output_mask_path = modality_output_dir / "mask.nii.gz"
                    sitk.WriteImage(processed_mask, str(output_mask_path))
                    print(f"    Mask saved to: {output_mask_path}")
                
                processing_results['modalities_processed'].append(modality)
                
            except Exception as e:
                error_msg = f"{modality} processing failed: {str(e)}"
                print(f"  Error: {error_msg}")
                print(f"  Traceback: {traceback.format_exc()}")
                processing_results['modalities_failed'].append({
                    'modality': modality,
                    'error': error_msg
                })
        
        # Check if any modality processed successfully
        if not processing_results['modalities_processed']:
            error_msg = f"All modalities failed for patient {patient_id}"
            if temp_output_dir.exists():
                shutil.rmtree(temp_output_dir)
            return {
                'patient_id': patient_id, 
                'status': 'error', 
                'error': error_msg,
                'failed_modalities': processing_results['modalities_failed']
            }
        
        # Move temporary directory to final location
        if final_output_dir.exists():
            shutil.rmtree(final_output_dir)
        final_output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temp_output_dir), str(final_output_dir))
        
        success_msg = f"Patient {patient_id} completed - Success: {processing_results['modalities_processed']}"
        if processing_results['modalities_failed']:
            success_msg += f" - Failed: {[m['modality'] for m in processing_results['modalities_failed']]}"
        
        print(f"\n{success_msg}")
        return {
            'patient_id': patient_id,
            'status': 'success',
            'processing_results': processing_results,
            'message': success_msg
        }
        
    except Exception as e:
        error_msg = f"Patient {patient_id} processing failed: {str(e)}"
        print(f"\nError: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)
        return {'patient_id': patient_id, 'status': 'error', 'error': error_msg}


class MRIPreprocessor:
    """MRI data preprocessing tool for resampling and alignment"""
    
    def __init__(self, base_data_dir, output_dir, target_spacing=(1.0, 1.0, 1.0)):
        self.base_data_dir = Path(base_data_dir)
        self.output_dir = Path(output_dir)
        self.target_spacing = target_spacing
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Progress and error log files
        self.progress_file = self.output_dir / "processing_progress.json"
        self.error_log_file = self.output_dir / "error_log.json"
        
        # Load existing progress
        self.processed_patients = self.load_progress()
        self.errors = self.load_errors()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "preprocessing.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_progress(self):
        """Load processing progress from file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_progress(self):
        """Save processing progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.processed_patients, f, indent=2)
    
    def load_errors(self):
        """Load error log from file"""
        if self.error_log_file.exists():
            with open(self.error_log_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_errors(self):
        """Save error log to file"""
        with open(self.error_log_file, 'w') as f:
            json.dump(self.errors, f, indent=2)
    
    def find_all_patients(self, datasets=None):
        """Find all patient folders in the dataset directories"""
        if datasets is None:
            datasets = [d.name for d in self.base_data_dir.iterdir() if d.is_dir()]
        
        all_patients = []
        for dataset in datasets:
            dataset_path = self.base_data_dir / dataset
            if dataset_path.exists():
                for patient_folder in dataset_path.iterdir():
                    if patient_folder.is_dir():
                        patient_id = f"{dataset}_{patient_folder.name}"
                        all_patients.append({
                            'patient_id': patient_id,
                            'dataset': dataset,
                            'path': patient_folder
                        })
        
        self.logger.info(f"Found {len(all_patients)} patients")
        return all_patients
    
    def run_preprocessing(self, max_workers=1):
        """Run the preprocessing pipeline"""
        self.logger.info(f"Starting preprocessing with {max_workers} worker(s)")
        self.logger.info(f"Target spacing: {self.target_spacing}")
        
        # Find all patients
        all_patients = self.find_all_patients()
        
        # Filter out already processed patients
        pending_patients = [p for p in all_patients if p['patient_id'] not in self.processed_patients]
        
        if not pending_patients:
            self.logger.info("All patients have been processed")
            self.print_summary()
            return
        
        self.logger.info(f"Processing {len(pending_patients)} patients")
        
        # Process patients with progress bar
        for patient in tqdm(pending_patients, desc="Processing patients"):
            result = process_single_patient_worker(
                patient, self.base_data_dir, self.output_dir, self.target_spacing
            )
            
            # Update records
            if result['status'] == 'success':
                self.processed_patients[result['patient_id']] = {
                    'timestamp': time.time(),
                    'processing_results': result['processing_results']
                }
                self.logger.info(f"✅ {result['patient_id']} processed successfully")
            else:
                self.errors[result['patient_id']] = result['error']
                self.logger.error(f"❌ {result['patient_id']} processing failed: {result['error']}")
            
            # Save progress in real-time
            self.save_progress()
            self.save_errors()
        
        # Clean up temporary directory
        temp_dir = self.output_dir / "temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print processing summary"""
        total_processed = len(self.processed_patients)
        total_errors = len(self.errors)
        
        print("\n" + "="*60)
        print("Processing Summary")
        print("="*60)
        print(f"✅ Successfully processed: {total_processed} patients")
        print(f"❌ Failed: {total_errors} patients")
        
        if self.processed_patients:
            print("\nModality statistics (successful patients):")
            modality_counts = {}
            for patient_id, info in self.processed_patients.items():
                processed_modalities = info['processing_results']['modalities_processed']
                for modality in processed_modalities:
                    modality_counts[modality] = modality_counts.get(modality, 0) + 1
            
            for modality, count in modality_counts.items():
                print(f"  {modality}: {count} patients")
        
        if self.errors:
            print("\nFailed patients:")
            for patient_id, error in list(self.errors.items())[:10]:
                print(f"  {patient_id}: {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")
        
        print(f"\n📁 Output directory: {self.output_dir}")
        print(f"📊 Progress file: {self.progress_file}")
        print(f"📋 Error log: {self.error_log_file}")


def main():
    """Main entry point"""
    print("🏥 MRI Data Preprocessing Tool")
    print("="*60)
    
    # Configuration
    base_data_dir = "./data/input"
    output_dir = "./data/output"
    target_spacing = (1.0, 1.0, 1.0)
    
    print(f"📂 Input directory: {base_data_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Target spacing: {target_spacing}")
    
    # Create preprocessor
    preprocessor = MRIPreprocessor(
        base_data_dir=base_data_dir,
        output_dir=output_dir,
        target_spacing=target_spacing
    )
    
    # Run preprocessing
    print("\n🚀 Starting preprocessing...\n")
    preprocessor.run_preprocessing(max_workers=1)
    
    print("\n✨ Preprocessing completed!")


if __name__ == "__main__":
    main()
