import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
import json
from tqdm import tqdm
import shutil
from datetime import datetime


class OrientationFixer:
    """Fix MRI image orientation to target coordinate system"""
    
    def __init__(self, data_dir, target_orientation='RAS'):
        self.data_dir = Path(data_dir)
        self.target_orientation = target_orientation
        self.backup_dir = self.data_dir.parent / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Log file
        self.log_file = self.data_dir.parent / "orientation_fix_log.txt"
        self.analysis_report = {}
    
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
        
        for item in self.data_dir.iterdir():
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
    
    def get_image_orientation(self, image_path):
        """Get image orientation information"""
        try:
            image = sitk.ReadImage(str(image_path))
            
            # Get orientation string from direction cosines
            orientation = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                image.GetDirection()
            )
            
            # Get detailed information
            direction_matrix = np.array(image.GetDirection()).reshape(3, 3)
            size = image.GetSize()
            spacing = image.GetSpacing()
            origin = image.GetOrigin()
            
            return {
                'orientation': orientation,
                'direction_matrix': direction_matrix.tolist(),
                'size': size,
                'spacing': spacing,
                'origin': origin,
                'success': True
            }
        except Exception as e:
            return {
                'orientation': 'UNKNOWN',
                'error': str(e),
                'success': False
            }
    
    def analyze_all_orientations(self):
        """Analyze orientation distribution across all patients"""
        self.log_message("Analyzing orientation distribution for all patients...")
        
        all_patients = self.find_all_patients()
        orientation_stats = {}
        patient_details = {}
        
        for patient_info in tqdm(all_patients, desc="Analyzing orientations"):
            patient_id = patient_info['patient_id']
            patient_path = patient_info['path']
            
            patient_modalities = {}
            patient_orientations = set()
            
            # Check each modality
            for modality in ['T1', 'T2', 'T1C']:
                ori_file = patient_path / modality / "ori.nii.gz"
                if ori_file.exists():
                    orientation_info = self.get_image_orientation(ori_file)
                    patient_modalities[modality] = orientation_info
                    
                    if orientation_info['success']:
                        patient_orientations.add(orientation_info['orientation'])
            
            # Record patient information
            if patient_modalities:
                # Use first successful modality as patient's main orientation
                main_orientation = None
                for modality_info in patient_modalities.values():
                    if modality_info['success']:
                        main_orientation = modality_info['orientation']
                        break
                
                patient_details[patient_id] = {
                    'main_orientation': main_orientation,
                    'all_orientations': list(patient_orientations),
                    'modalities': patient_modalities,
                    'needs_fix': main_orientation != self.target_orientation,
                    'consistent_within_patient': len(patient_orientations) == 1
                }
                
                # Collect orientation statistics
                if main_orientation not in orientation_stats:
                    orientation_stats[main_orientation] = []
                orientation_stats[main_orientation].append(patient_id)
        
        # Save analysis results
        self.analysis_report = {
            'orientation_stats': orientation_stats,
            'patient_details': patient_details,
            'target_orientation': self.target_orientation,
            'analysis_time': datetime.now().isoformat()
        }
        
        # Print statistics
        self.log_message(f"Analysis complete: {len(patient_details)} patients checked")
        self.log_message("Orientation distribution:")
        
        total_patients = len(patient_details)
        needs_fix_count = 0
        
        for orientation, patients in orientation_stats.items():
            count = len(patients)
            percentage = count / total_patients * 100
            is_target = "✅" if orientation == self.target_orientation else "❌"
            self.log_message(f"  {is_target} {orientation}: {count} patients ({percentage:.1f}%)")
            
            if orientation != self.target_orientation:
                needs_fix_count += count
                self.log_message(f"    Example patients needing fix: {patients[:3]}")
        
        self.log_message(f"Patients needing fix: {needs_fix_count}")
        
        # Check consistency within patients
        inconsistent_patients = [
            pid for pid, details in patient_details.items()
            if not details['consistent_within_patient']
        ]
        
        if inconsistent_patients:
            self.log_message(f"⚠️  Found {len(inconsistent_patients)} patients with inconsistent orientations:")
            for pid in inconsistent_patients[:5]:
                orientations = patient_details[pid]['all_orientations']
                self.log_message(f"    {pid}: {orientations}")
        
        return orientation_stats, patient_details
    
    def backup_patient_data(self, patient_path, patient_id):
        """Backup patient data before modification"""
        try:
            backup_patient_dir = self.backup_dir / patient_id
            backup_patient_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup all modalities
            for modality in ['T1', 'T2', 'T1C']:
                modality_path = patient_path / modality
                if modality_path.exists():
                    backup_modality_dir = backup_patient_dir / modality
                    shutil.copytree(modality_path, backup_modality_dir, dirs_exist_ok=True)
            
            return True
        except Exception as e:
            self.log_message(f"Backup failed for patient {patient_id}: {e}")
            return False
    
    def reorient_image(self, image_path, target_orientation):
        """Reorient image to target orientation"""
        try:
            # Read image
            image = sitk.ReadImage(str(image_path))
            
            # Check current orientation
            current_orientation = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                image.GetDirection()
            )
            
            if current_orientation == target_orientation:
                return image, current_orientation, target_orientation, False  # No fix needed
            
            # Use DICOMOrientImageFilter to reorient
            orient_filter = sitk.DICOMOrientImageFilter()
            orient_filter.SetDesiredCoordinateOrientation(target_orientation)
            
            reoriented_image = orient_filter.Execute(image)
            
            # Verify result
            final_orientation = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                reoriented_image.GetDirection()
            )
            
            return reoriented_image, current_orientation, final_orientation, True
            
        except Exception as e:
            raise Exception(f"Reorientation failed: {e}")
    
    def fix_patient_orientation(self, patient_path, patient_id, dry_run=True):
        """Fix orientation for a single patient"""
        fix_results = {
            'patient_id': patient_id,
            'modalities_processed': [],
            'modalities_failed': [],
            'backup_created': False,
            'changes_made': False
        }
        
        try:
            # Check if fix is needed
            needs_fix = False
            for modality in ['T1', 'T2', 'T1C']:
                ori_file = patient_path / modality / "ori.nii.gz"
                if ori_file.exists():
                    orientation_info = self.get_image_orientation(ori_file)
                    if orientation_info['success'] and orientation_info['orientation'] != self.target_orientation:
                        needs_fix = True
                        break
            
            if not needs_fix:
                self.log_message(f"  {patient_id}: Already correct orientation, skipped")
                return fix_results
            
            # Create backup (only for actual fix, not dry run)
            if not dry_run:
                if self.backup_patient_data(patient_path, patient_id):
                    fix_results['backup_created'] = True
                    self.log_message(f"  {patient_id}: Backup created")
                else:
                    raise Exception("Backup creation failed")
            
            # Process each modality
            for modality in ['T1', 'T2', 'T1C']:
                modality_path = patient_path / modality
                ori_file = modality_path / "ori.nii.gz"
                mask_file = modality_path / "mask.nii.gz"
                
                if ori_file.exists():
                    try:
                        # Reorient original image
                        reoriented_image, current_orient, final_orient, changed = self.reorient_image(
                            ori_file, self.target_orientation
                        )
                        
                        if changed:
                            if not dry_run:
                                # Save reoriented image
                                sitk.WriteImage(reoriented_image, str(ori_file))
                            
                            fix_results['changes_made'] = True
                            self.log_message(f"    {modality}: {current_orient} -> {final_orient}")
                            
                            # Process mask
                            if mask_file.exists():
                                try:
                                    reoriented_mask, _, _, _ = self.reorient_image(
                                        mask_file, self.target_orientation
                                    )
                                    if not dry_run:
                                        sitk.WriteImage(reoriented_mask, str(mask_file))
                                    self.log_message(f"    {modality} mask: Reoriented")
                                except Exception as e:
                                    self.log_message(f"    {modality} mask reorientation failed: {e}")
                        else:
                            self.log_message(f"    {modality}: Already correct orientation")
                        
                        fix_results['modalities_processed'].append(modality)
                        
                    except Exception as e:
                        error_msg = f"{modality} processing failed: {e}"
                        self.log_message(f"    {error_msg}")
                        fix_results['modalities_failed'].append({
                            'modality': modality,
                            'error': error_msg
                        })
            
            if fix_results['changes_made']:
                action = "dry-run" if dry_run else "fixed"
                self.log_message(f"  {patient_id}: {action}")
            
            return fix_results
            
        except Exception as e:
            error_msg = f"Patient {patient_id} processing failed: {e}"
            self.log_message(error_msg)
            fix_results['error'] = error_msg
            return fix_results
    
    def fix_all_orientations(self, dry_run=True):
        """Fix orientations for all patients needing correction"""
        if not hasattr(self, 'analysis_report') or not self.analysis_report:
            self.log_message("Please run analysis first")
            return
        
        action = "dry-run" if dry_run else "fixing"
        self.log_message(f"Starting {action} all patient orientations...")
        
        if not dry_run:
            # Create backup directory
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            self.log_message(f"Backup directory: {self.backup_dir}")
        
        patient_details = self.analysis_report['patient_details']
        
        # Filter patients needing fix
        patients_to_fix = [
            pid for pid, details in patient_details.items()
            if details['needs_fix']
        ]
        
        self.log_message(f"Patients needing {action}: {len(patients_to_fix)}")
        
        if not patients_to_fix:
            self.log_message("No patients need fixing")
            return
        
        # Fix patients
        fix_results = []
        successful_fixes = 0
        
        all_patients = self.find_all_patients()
        patient_path_map = {p['patient_id']: p['path'] for p in all_patients}
        
        for patient_id in tqdm(patients_to_fix, desc=f"Processing ({action})"):
            if patient_id in patient_path_map:
                result = self.fix_patient_orientation(
                    patient_path_map[patient_id], 
                    patient_id, 
                    dry_run
                )
                fix_results.append(result)
                
                if result.get('changes_made', False) and not result.get('error'):
                    successful_fixes += 1
        
        # Save fix report
        fix_report = {
            'fix_type': action,
            'target_orientation': self.target_orientation,
            'total_patients_to_fix': len(patients_to_fix),
            'successful_fixes': successful_fixes,
            'fix_results': fix_results,
            'fix_time': datetime.now().isoformat()
        }
        
        report_file = self.data_dir.parent / f"fix_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(fix_report, f, indent=2, ensure_ascii=False)
        
        self.log_message(f"{action.capitalize()} complete!")
        self.log_message(f"Successful: {successful_fixes}/{len(patients_to_fix)} patients")
        self.log_message(f"Detailed report: {report_file}")
        
        if not dry_run and successful_fixes > 0:
            self.log_message(f"Backup location: {self.backup_dir}")
    
    def create_visualization_report(self):
        """Create visualization report of orientation analysis"""
        if not hasattr(self, 'analysis_report') or not self.analysis_report:
            self.log_message("Please run analysis first")
            return
        
        orientation_stats = self.analysis_report['orientation_stats']
        patient_details = self.analysis_report['patient_details']
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Orientation distribution pie chart
        orientations = list(orientation_stats.keys())
        counts = [len(patients) for patients in orientation_stats.values()]
        colors = ['lightgreen' if o == self.target_orientation else 'lightcoral' for o in orientations]
        
        ax1.pie(counts, labels=orientations, autopct='%1.1f%%', colors=colors)
        ax1.set_title('Image Orientation Distribution', fontsize=12, fontweight='bold')
        
        # 2. Fix requirements bar chart
        needs_fix = sum(1 for d in patient_details.values() if d['needs_fix'])
        no_fix_needed = len(patient_details) - needs_fix
        
        ax2.bar(['Needs Fix', 'Already Correct'], [needs_fix, no_fix_needed], 
                color=['lightcoral', 'lightgreen'])
        ax2.set_title('Fix Requirements', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Patients')
        
        # 3. Detailed statistics table
        ax3.axis('off')
        table_data = []
        for orientation, patients in orientation_stats.items():
            count = len(patients)
            percentage = count / len(patient_details) * 100
            status = "✅ Target" if orientation == self.target_orientation else "❌ Needs Fix"
            table_data.append([orientation, count, f"{percentage:.1f}%", status])
        
        table = ax3.table(cellText=table_data,
                         colLabels=['Orientation', 'Count', 'Percentage', 'Status'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax3.set_title('Detailed Statistics', fontsize=12, fontweight='bold')
        
        # 4. Problem patients list
        ax4.axis('off')
        problem_patients = [pid for pid, details in patient_details.items() if details['needs_fix']]
        
        if problem_patients:
            problem_text = "Patients Needing Orientation Fix:\n\n"
            for i, pid in enumerate(problem_patients[:15]):
                orientation = patient_details[pid]['main_orientation']
                problem_text += f"{i+1:2d}. {pid} ({orientation})\n"
            
            if len(problem_patients) > 15:
                problem_text += f"\n... and {len(problem_patients) - 15} more patients"
            
            ax4.text(0.1, 0.9, problem_text, transform=ax4.transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
        else:
            ax4.text(0.5, 0.5, "No patients need orientation fix!", 
                    transform=ax4.transAxes, ha='center', va='center',
                    fontsize=14, color='green', fontweight='bold')
        
        ax4.set_title('Problem Patients List', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save report
        viz_file = self.data_dir.parent / f"orientation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log_message(f"Visualization report saved: {viz_file}")
        return viz_file


def main():
    """Main entry point"""
    print("🔧 MRI Image Orientation Fixer")
    print("="*60)
    
    # Configuration
    data_dir = "./data/output"
    target_orientation = "RAS"
    
    print(f"📂 Data directory: {data_dir}")
    print(f"🎯 Target orientation: {target_orientation}")
    
    # Create fixer
    fixer = OrientationFixer(data_dir, target_orientation)
    
    # Step 1: Analyze orientations
    print("\n🔍 Step 1: Analyzing orientation distribution...")
    orientation_stats, patient_details = fixer.analyze_all_orientations()
    
    # Save analysis report
    analysis_file = fixer.data_dir.parent / f"orientation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(fixer.analysis_report, f, indent=2, ensure_ascii=False)
    print(f"📊 Analysis report saved: {analysis_file}")
    
    # Step 2: Dry run
    needs_fix = sum(1 for d in patient_details.values() if d['needs_fix'])
    
    if needs_fix == 0:
        print("\n✅ All patients already have correct orientation!")
        return
    
    print(f"\n🎭 Step 2: Dry-run fix for {needs_fix} patients...")
    fixer.fix_all_orientations(dry_run=True)
    
    # Step 3: Create visualization
    print("\n📊 Step 3: Creating visualization report...")
    fixer.create_visualization_report()
    
    # Step 4: Ask for confirmation
    print("\n⚠️  Step 4: Ready for actual fix")
    print(f"⚠️  This will permanently modify {needs_fix} patient images")
    print("⚠️  Backups will be created before modification")
    
    confirm = input("\nProceed with actual orientation fix? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Cancelled")
        return
    
    # Step 5: Execute actual fix
    print("\n🔧 Step 5: Executing actual orientation fix...")
    fixer.fix_all_orientations(dry_run=False)
    
    print("\n✨ Orientation fix completed!")


if __name__ == "__main__":
    main()
