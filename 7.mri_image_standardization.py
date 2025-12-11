"""
Image Preprocessing and Standardization Tool

Purpose:
    Batch process medical images to standardize dimensions
    Support padding for small images and cropping for large images
    Maintain image center alignment during processing

Features:
    - Resize images to target dimensions (default: 512x512)
    - Center-crop images larger than target size
    - Center-pad images smaller than target size
    - Incremental processing (skip already processed files)
    - Comprehensive error handling and logging
    - Progress tracking with detailed statistics

Processing Logic:
    1. Scan all image files in source directory
    2. Check which files have been processed
    3. Process remaining images:
       - If size > target: center crop
       - If size < target: center pad with zero values
    4. Generate processing report with statistics

Supported Formats: PNG, JPG, JPEG, BMP, TIF

Usage:
    python mri_image_standardization.py
    
    Configure paths and parameters in the CONFIG section below.

Author: MRI Processing Team
Version: 1.0
License: MIT
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional


# ==================== Configuration ====================

CONFIG = {
    'source_root': './data/raw_images',
    'output_root': './data/standardized',
    'target_size': (512, 512),  # (height, width)
    'padding_value': 0,  # Value used for padding (black)
    'supported_formats': ('.png', '.jpg', '.jpeg', '.bmp', '.tif'),
}


# ==================== Image Processing Functions ====================

def process_image_to_target(
    image_path: str,
    target_size: Tuple[int, int] = (512, 512),
    padding_value: int = 0
) -> Optional[np.ndarray]:
    """
    Read and process image to target size.
    
    Processing strategy:
    - Larger than target: center crop
    - Smaller than target: center pad
    - Equal to target: return as-is
    
    Args:
        image_path: Path to input image file
        target_size: Target dimensions (height, width)
        padding_value: Value used for padding (default: 0 for black)
        
    Returns:
        Processed image as numpy array, or None if processing failed
    """
    try:
        # Read image in grayscale mode
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            # Image read failed
            if os.path.exists(image_path):
                file_size = os.path.getsize(image_path)
                if file_size == 0:
                    print(f"[WARNING] Empty file: {image_path}")
                else:
                    print(f"[WARNING] Cannot read image: {image_path} ({file_size} bytes)")
            else:
                print(f"[WARNING] File not found: {image_path}")
            return None
        
        h, w = img.shape
        target_h, target_w = target_size
        
        # Case 1: Image larger than target - center crop
        if h > target_h or w > target_w:
            start_y = max(0, (h - target_h) // 2)
            start_x = max(0, (w - target_w) // 2)
            end_y = min(h, start_y + target_h)
            end_x = min(w, start_x + target_w)
            
            cropped_img = img[start_y:end_y, start_x:end_x]
            crop_h, crop_w = cropped_img.shape
            
            # Handle edge cases - if crop is still not target size, pad it
            if crop_h < target_h or crop_w < target_w:
                pad_top = (target_h - crop_h) // 2
                pad_bottom = target_h - crop_h - pad_top
                pad_left = (target_w - crop_w) // 2
                pad_right = target_w - crop_w - pad_left
                
                processed_img = cv2.copyMakeBorder(
                    cropped_img, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=padding_value
                )
            else:
                processed_img = cropped_img
        
        # Case 2: Image smaller or equal to target - center pad
        else:
            pad_top = (target_h - h) // 2
            pad_bottom = target_h - h - pad_top
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left
            
            processed_img = cv2.copyMakeBorder(
                img, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=padding_value
            )
        
        return processed_img
        
    except Exception as e:
        print(f"[ERROR] Exception processing image {image_path}: {e}")
        return None


def verify_processed_file(
    output_path: str,
    target_size: Tuple[int, int]
) -> bool:
    """
    Verify if output file is valid and has correct dimensions.
    
    Args:
        output_path: Path to output image file
        target_size: Expected image dimensions
        
    Returns:
        True if file exists and is valid, False otherwise
    """
    if not os.path.exists(output_path):
        return False
    
    try:
        # Check file is not empty
        if os.path.getsize(output_path) == 0:
            return False
        
        # Verify image can be read and has correct size
        test_img = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
        if test_img is None:
            return False
        
        if test_img.shape != target_size:
            return False
        
        return True
        
    except Exception:
        return False


# ==================== File Discovery Functions ====================

def collect_all_image_paths(source_root: str) -> List[str]:
    """
    Scan directory tree and collect all image file paths.
    
    Looks for 'ori' and 'mask' subdirectories and collects
    all supported image formats within them.
    
    Args:
        source_root: Root directory to scan
        
    Returns:
        List of image file paths
    """
    print("[INFO] Step 1: Scanning for image files...")
    
    all_paths = []
    target_dirs = []
    
    # First, find all 'ori' and 'mask' directories
    for dirpath, _, _ in os.walk(source_root):
        dir_name = os.path.basename(dirpath)
        if dir_name in ['ori', 'mask']:
            target_dirs.append(dirpath)
    
    print(f"        Found {len(target_dirs)} target directories")
    
    # Scan each directory for image files
    for dirpath in tqdm(target_dirs, desc="Scanning directories"):
        try:
            filenames = os.listdir(dirpath)
            for filename in filenames:
                if filename.lower().endswith(CONFIG['supported_formats']):
                    all_paths.append(os.path.join(dirpath, filename))
        except Exception as e:
            print(f"[WARNING] Cannot read directory {dirpath}: {e}")
    
    print(f"        Total images found: {len(all_paths)}")
    return all_paths


def check_processing_status(
    image_paths: List[str],
    source_root: str,
    output_root: str,
    target_size: Tuple[int, int]
) -> Tuple[List[str], int]:
    """
    Check which files have been processed and need processing.
    
    Args:
        image_paths: List of source image paths
        source_root: Source root directory
        output_root: Output root directory
        target_size: Expected output dimensions
        
    Returns:
        Tuple of (paths_to_process, already_processed_count)
    """
    print("[INFO] Step 2: Checking processing status...")
    
    to_process = []
    already_processed = 0
    
    for input_path in tqdm(image_paths, desc="Checking status"):
        output_path = input_path.replace(source_root, output_root)
        
        if verify_processed_file(output_path, target_size):
            already_processed += 1
        else:
            to_process.append(input_path)
    
    print(f"        Need processing: {len(to_process)}")
    print(f"        Already processed: {already_processed}")
    
    return to_process, already_processed


# ==================== Main Processing Function ====================

def process_images(
    image_paths_to_process: List[str],
    source_root: str,
    output_root: str,
    target_size: Tuple[int, int],
    padding_value: int = 0
) -> Tuple[List[str], int]:
    """
    Process all images to target size.
    
    Args:
        image_paths_to_process: List of image paths to process
        source_root: Source root directory
        output_root: Output root directory
        target_size: Target image dimensions
        padding_value: Value for padding (default: 0)
        
    Returns:
        Tuple of (failed_files_list, success_count)
    """
    if not image_paths_to_process:
        print("[INFO] No files to process")
        return [], 0
    
    print("[INFO] Step 3: Processing images...")
    
    failed_files = []
    success_count = 0
    
    for img_path in tqdm(image_paths_to_process, desc="Processing", unit="images"):
        filename = os.path.basename(img_path)
        
        # Process image
        processed_img = process_image_to_target(
            img_path, target_size, padding_value
        )
        
        if processed_img is not None:
            # Create output path and directory
            output_path = img_path.replace(source_root, output_root)
            output_folder = os.path.dirname(output_path)
            
            try:
                os.makedirs(output_folder, exist_ok=True)
                
                # Save processed image
                success = cv2.imwrite(output_path, processed_img)
                if success:
                    success_count += 1
                    tqdm.write(f"[OK] {filename}")
                else:
                    tqdm.write(f"[FAIL] Cannot save: {filename}")
                    failed_files.append(img_path)
                    
            except Exception as e:
                tqdm.write(f"[ERROR] Save exception for {filename}: {e}")
                failed_files.append(img_path)
        else:
            tqdm.write(f"[FAIL] Cannot process: {filename}")
            failed_files.append(img_path)
    
    return failed_files, success_count


# ==================== Analysis and Reporting ====================

def analyze_failed_files(failed_files: List[str], sample_size: int = 10) -> None:
    """
    Analyze and report on failed files.
    
    Args:
        failed_files: List of failed file paths
        sample_size: Number of files to analyze in detail
    """
    if not failed_files:
        return
    
    print("\n[INFO] Failed File Analysis")
    print(f"        Total failed: {len(failed_files)}")
    
    # Analyze sample of failed files
    analysis_count = min(sample_size, len(failed_files))
    
    for file_path in tqdm(failed_files[:analysis_count], desc="Analyzing"):
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            tqdm.write(f"  {os.path.basename(file_path)}: {file_size} bytes")
            
            # Try alternative read methods
            try:
                from PIL import Image
                pil_img = Image.open(file_path)
                tqdm.write(f"    -> PIL readable: {pil_img.size} {pil_img.mode}")
            except Exception:
                tqdm.write(f"    -> PIL unreadable")
        else:
            tqdm.write(f"  [MISSING] {file_path}")
    
    if len(failed_files) > sample_size:
        print(f"        ... and {len(failed_files) - sample_size} more")


def generate_report(
    total_images: int,
    success_count: int,
    failed_count: int,
    already_processed: int,
    output_root: str,
    target_size: Tuple[int, int]
) -> None:
    """
    Generate and display processing report.
    
    Args:
        total_images: Total images found
        success_count: Successfully processed count
        failed_count: Failed files count
        already_processed: Already processed count
        output_root: Output directory
        target_size: Target image dimensions
    """
    print("\n" + "="*70)
    print("PROCESSING COMPLETE - SUMMARY REPORT")
    print("="*70)
    print(f"Configuration:")
    print(f"  Target size: {target_size[0]} x {target_size[1]} pixels")
    print(f"  Output directory: {output_root}")
    print(f"\nStatistics:")
    print(f"  Total images found: {total_images}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Already processed (skipped): {already_processed}")
    
    if total_images > 0:
        success_rate = (success_count / (total_images - already_processed)) * 100 if (total_images - already_processed) > 0 else 0
        print(f"\nSuccess rate: {success_rate:.1f}%")
    
    print("="*70)


def process_all_data(
    source_root: str,
    output_root: str,
    target_size: Tuple[int, int] = (512, 512),
    padding_value: int = 0
) -> None:
    """
    Main processing pipeline.
    
    Orchestrates all steps:
    1. Collect image paths
    2. Check processing status
    3. Process images
    4. Generate report
    
    Args:
        source_root: Root directory containing raw images
        output_root: Root directory for processed images
        target_size: Target image dimensions (height, width)
        padding_value: Value for padding operations
    """
    print("\n" + "="*70)
    print("IMAGE PREPROCESSING AND STANDARDIZATION")
    print("="*70)
    print(f"Configuration:")
    print(f"  Source root: {source_root}")
    print(f"  Output root: {output_root}")
    print(f"  Target size: {target_size[0]} x {target_size[1]} pixels")
    print("="*70 + "\n")
    
    # Validate paths
    if not os.path.isdir(source_root):
        print(f"[ERROR] Source directory not found: {source_root}")
        return
    
    # Step 1: Collect all image paths
    all_image_paths = collect_all_image_paths(source_root)
    
    if not all_image_paths:
        print("[ERROR] No image files found. Please check source directory.")
        return
    
    # Step 2: Check processing status
    paths_to_process, already_processed = check_processing_status(
        all_image_paths, source_root, output_root, target_size
    )
    
    # Step 3: Process images
    failed_files, success_count = process_images(
        paths_to_process, source_root, output_root, target_size, padding_value
    )
    
    # Step 4: Analyze failures (if any)
    if failed_files:
        analyze_failed_files(failed_files)
    
    # Step 5: Generate report
    generate_report(
        len(all_image_paths),
        success_count,
        len(failed_files),
        already_processed,
        output_root,
        target_size
    )


# ==================== Entry Point ====================

if __name__ == '__main__':
    # Use configuration from CONFIG dict
    process_all_data(
        source_root=CONFIG['source_root'],
        output_root=CONFIG['output_root'],
        target_size=CONFIG['target_size'],
        padding_value=CONFIG['padding_value']
    )
