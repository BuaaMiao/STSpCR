import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import json
import warnings

warnings.filterwarnings('ignore')


class AdvancedMultiModalRegistration:
    """
    Advanced multi-modal MRI image registration using multiple methods and fusion.
    
    Features:
    - Multiple registration methods (ECC, phase correlation, feature-based)
    - Multi-scale similarity computation
    - Global and local registration in two stages
    - Automatic quality control and refinement
    - Works with PNG 2D slices instead of 3D NIfTI files
    """
    
    def __init__(self, source_root, target_root, skip_existing=True):
        """Initialize advanced registration engine
        
        Args:
            source_root: Source data directory
            target_root: Target output directory
            skip_existing: Skip already processed patients
        """
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        self.target_root.mkdir(exist_ok=True)
        self.skip_existing = skip_existing
        
        # Registration parameters
        self.similarity_threshold = 0.06
        self.max_layer_diff_ratio = 0.9
        self.min_correspondence_ratio = 0.1
        
        # Multi-stage parameters
        self.global_registration_samples = 12
        self.transform_smoothing_sigma = 1.5
        self.max_transform_change = 4.0
        
        # Multi-scale parameters
        self.use_multiscale = True
        self.pyramid_levels = 3
        self.scales = [1.0, 0.5, 0.25]
        self.scale_weights = [0.6, 0.3, 0.1]
        
        # Quality control parameters
        self.quality_threshold = 0.15
        self.outlier_rejection_ratio = 0.1
        self.min_registration_quality = 0.08
        
        # Method weights
        self.method_weights = {
            'phase_correlation': 1.0,
            'ecc_translation': 0.8,
            'ecc_euclidean': 0.6,
            'feature_orb': 0.4,
        }
    
    def preprocess_image(self, img):
        """Preprocess image with CLAHE and filtering
        
        Args:
            img: Input image array (0-1 range)
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to uint8 for CLAHE
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            
            # Adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_eq = clahe.apply(img_uint8).astype(np.float32) / 255.0
            
            # Edge-preserving bilateral filtering
            img_filtered = cv2.bilateralFilter(
                (img_eq * 255).astype(np.uint8), 9, 50, 50
            ).astype(np.float32) / 255.0
            
            # Adaptive contrast enhancement
            mean_intensity = np.mean(img_filtered)
            alpha = 1.0 + (0.5 - mean_intensity) * 0.5
            img_enhanced = cv2.convertScaleAbs(
                img_filtered * 255, alpha=alpha, beta=10
            ).astype(np.float32) / 255.0
            
            return np.clip(img_enhanced, 0, 1)
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return np.clip(img, 0, 1)
    
    def _calculate_ncc(self, img1, img2):
        """Calculate normalized cross-correlation
        
        Args:
            img1, img2: Input images
            
        Returns:
            NCC value (-1 to 1)
        """
        try:
            mean1, mean2 = np.mean(img1), np.mean(img2)
            std1, std2 = np.std(img1), np.std(img2)
            
            if std1 > 1e-6 and std2 > 1e-6:
                ncc = np.mean((img1 - mean1) * (img2 - mean2)) / (std1 * std2)
                return max(-1, min(1, ncc))
            return 0.0
        except:
            return 0.0
    
    def _calculate_ssim(self, img1, img2):
        """Calculate structural similarity index
        
        Args:
            img1, img2: Input images
            
        Returns:
            SSIM value (0 to 1)
        """
        try:
            mu1, mu2 = np.mean(img1), np.mean(img2)
            sigma1_sq = np.var(img1)
            sigma2_sq = np.var(img2)
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
            
            c1, c2 = (0.01)**2, (0.03)**2
            ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / \
                   ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
            
            return max(0, min(1, ssim))
        except:
            return 0.0
    
    def _calculate_gradient_correlation(self, img1, img2):
        """Calculate gradient correlation
        
        Args:
            img1, img2: Input images
            
        Returns:
            Gradient correlation (0 to 1)
        """
        try:
            grad1_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
            grad1_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
            grad2_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
            grad2_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
            
            grad1_mag = np.sqrt(grad1_x**2 + grad1_y**2)
            grad2_mag = np.sqrt(grad2_x**2 + grad2_y**2)
            
            if np.std(grad1_mag) > 1e-6 and np.std(grad2_mag) > 1e-6:
                corr = np.corrcoef(grad1_mag.flatten(), grad2_mag.flatten())[0, 1]
                return 0 if np.isnan(corr) else max(0, corr)
            return 0.0
        except:
            return 0.0
    
    def _calculate_mi_optimized(self, img1, img2):
        """Calculate optimized mutual information
        
        Args:
            img1, img2: Input images (0-1 range)
            
        Returns:
            Normalized mutual information
        """
        try:
            img1_q = (img1 * 127).astype(np.uint8)
            img2_q = (img2 * 127).astype(np.uint8)
            
            # Joint histogram
            hist_2d, _, _ = np.histogram2d(
                img1_q.flatten(), img2_q.flatten(), 
                bins=64, range=[[0, 127], [0, 127]]
            )
            
            # Smooth histogram
            hist_2d = gaussian_filter(hist_2d, sigma=0.5)
            hist_2d = hist_2d + 1e-10
            hist_2d = hist_2d / np.sum(hist_2d)
            
            # Marginal distributions
            hist1 = np.sum(hist_2d, axis=1)
            hist2 = np.sum(hist_2d, axis=0)
            
            # Mutual information
            mi = np.sum(hist_2d * np.log(hist_2d / np.outer(hist1, hist2)))
            
            # Normalization
            h1 = -np.sum(hist1 * np.log(hist1))
            h2 = -np.sum(hist2 * np.log(hist2))
            
            return mi / max(h1, h2) if max(h1, h2) > 0 else 0
        except:
            return 0.0
    
    def calculate_multiscale_similarity(self, img1, img2):
        """Calculate multi-scale similarity
        
        Args:
            img1, img2: Input images
            
        Returns:
            Combined similarity score (0 to 1)
        """
        if not self.use_multiscale:
            return self.calculate_single_scale_similarity(img1, img2, 1.0)
        
        try:
            similarities = []
            
            for scale, weight in zip(self.scales, self.scale_weights):
                if scale < 1.0:
                    h, w = img1.shape
                    new_h, new_w = max(32, int(h * scale)), max(32, int(w * scale))
                    img1_scaled = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    img2_scaled = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    img1_scaled, img2_scaled = img1, img2
                
                similarity = self.calculate_single_scale_similarity(img1_scaled, img2_scaled, scale)
                similarities.append(similarity * weight)
            
            final_similarity = sum(similarities) / sum(self.scale_weights)
            return max(0, min(1, final_similarity))
            
        except Exception as e:
            print(f"Multi-scale similarity error: {e}")
            return self.calculate_single_scale_similarity(img1, img2, 1.0)
    
    def calculate_single_scale_similarity(self, img1, img2, scale=1.0):
        """Calculate single-scale similarity
        
        Args:
            img1, img2: Input images
            scale: Scale factor
            
        Returns:
            Similarity score (0 to 1)
        """
        try:
            img1_proc = self.preprocess_image(img1)
            img2_proc = self.preprocess_image(img2)
            
            # Calculate similarity metrics
            ncc = self._calculate_ncc(img1_proc, img2_proc)
            ssim = self._calculate_ssim(img1_proc, img2_proc)
            mi = self._calculate_mi_optimized(img1_proc, img2_proc)
            grad_corr = self._calculate_gradient_correlation(img1_proc, img2_proc)
            
            # Adjust weights by scale
            if scale >= 1.0:
                weights = [0.2, 0.3, 0.3, 0.2]
            else:
                weights = [0.3, 0.2, 0.4, 0.1]
            
            combined = (weights[0] * max(0, ncc) + 
                       weights[1] * max(0, ssim) + 
                       weights[2] * max(0, mi) + 
                       weights[3] * max(0, grad_corr))
            
            return max(0, min(1, combined))
            
        except Exception as e:
            print(f"Single-scale similarity error: {e}")
            return 0.0
    
    def save_registration_results(self, results, output_dir, stats=None):
        """Save registration results to disk
        
        Args:
            results: Dictionary of registration results
            output_dir: Output directory path
            stats: Registration statistics
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for modality, data in results.items():
            modality_dir = output_dir / modality
            modality_dir.mkdir(exist_ok=True)
            
            # Save ori images
            ori_dir = modality_dir / "ori"
            ori_dir.mkdir(exist_ok=True)
            
            for i, (img, idx) in enumerate(zip(data['ori_stack'], data['ori_indices'])):
                img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                filename = ori_dir / f"ori_{idx:03d}.png"
                cv2.imwrite(str(filename), img_uint8)
            
            # Save mask images
            if data['mask_stack'] is not None:
                mask_dir = modality_dir / "mask"
                mask_dir.mkdir(exist_ok=True)
                
                for i, (img, idx) in enumerate(zip(data['mask_stack'], data['mask_indices'])):
                    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                    filename = mask_dir / f"mask_{idx:03d}.png"
                    cv2.imwrite(str(filename), img_uint8)
            
            # Save registration info
            info = {
                'modality': modality,
                'num_slices': len(data['ori_indices']),
                'indices': [int(x) for x in data['ori_indices']],
            }
            
            # Add registration statistics
            if 'success_rate' in data:
                info['success_rate'] = float(data['success_rate'])
            if 'avg_quality' in data:
                info['avg_quality'] = float(data['avg_quality'])
            if 'quality_scores' in data:
                info['quality_scores'] = [float(q) for q in data['quality_scores']]
            if 'global_transform' in data:
                info['global_transform'] = data['global_transform'].tolist()
            
            info_file = modality_dir / "registration_info.json"
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
        
        # Save overall statistics
        if stats:
            stats_file = output_dir / "registration_summary.json"
            with open(stats_file, 'w') as f:
                summary = {
                    'total_modalities_attempted': stats['total_attempts'] + 1,
                    'successful_registrations': stats['successful_registrations'],
                    'overall_success_rate': stats['successful_registrations'] / (stats['total_attempts'] + 1),
                    'average_quality': float(np.mean(stats['quality_scores'])) if stats['quality_scores'] else 0,
                    'quality_std': float(np.std(stats['quality_scores'])) if stats['quality_scores'] else 0,
                    'quality_distribution': {
                        'min': float(np.min(stats['quality_scores'])) if stats['quality_scores'] else 0,
                        'max': float(np.max(stats['quality_scores'])) if stats['quality_scores'] else 0,
                        'median': float(np.median(stats['quality_scores'])) if stats['quality_scores'] else 0,
                    }
                }
                json.dump(summary, f, indent=2)
    
    def multi_method_registration(self, ref_img, mov_img):
        """Multi-method registration fusion
        
        Args:
            ref_img: Reference image
            mov_img: Moving image
            
        Returns:
            Tuple of (transform_matrix, quality)
        """
        results = []
        
        for method, weight in self.method_weights.items():
            try:
                transform, quality = self.register_single_pair(ref_img, mov_img, method)
                if quality > self.min_registration_quality:
                    results.append((transform, quality * weight, method))
            except Exception as e:
                print(f"Method {method} failed: {e}")
                continue
        
        if not results:
            print("All registration methods failed")
            return np.eye(2, 3, dtype=np.float32), 0.0
        
        if len(results) == 1:
            return results[0][0], results[0][1] / self.method_weights[results[0][2]]
        
        return self.fuse_registration_results(results, ref_img, mov_img)
    
    def fuse_registration_results(self, results, ref_img, mov_img):
        """Fuse multiple registration results
        
        Args:
            results: List of (transform, quality, method) tuples
            ref_img: Reference image
            mov_img: Moving image
            
        Returns:
            Tuple of (fused_transform, fused_quality)
        """
        results.sort(key=lambda x: x[1], reverse=True)
        
        if len(results) < 2 or results[0][1] > results[1][1] * 1.8:
            best_method = results[0][2]
            return results[0][0], results[0][1] / self.method_weights[best_method]
        
        transform1, quality1, method1 = results[0]
        transform2, quality2, method2 = results[1]
        
        quality1_norm = quality1 / self.method_weights[method1]
        quality2_norm = quality2 / self.method_weights[method2]
        
        total_quality = quality1_norm + quality2_norm
        w1 = quality1_norm / total_quality
        w2 = quality2_norm / total_quality
        
        fused_transform = w1 * transform1 + w2 * transform2
        
        try:
            registered = cv2.warpAffine(mov_img, fused_transform, ref_img.shape[::-1])
            fused_quality = self.calculate_multiscale_similarity(ref_img, registered)
            
            if fused_quality < quality1_norm * 0.9:
                return transform1, quality1_norm
            
            print(f"Fused registration: {method1}+{method2}, quality: {fused_quality:.3f}")
            return fused_transform, fused_quality
            
        except Exception as e:
            print(f"Fusion failed: {e}")
            return results[0][0], quality1_norm
    
    def register_single_pair(self, img_fixed, img_moving, method='phase_correlation'):
        """Register single image pair with specified method
        
        Args:
            img_fixed: Fixed (reference) image
            img_moving: Moving image
            method: Registration method name
            
        Returns:
            Tuple of (transform_matrix, quality)
        """
        if method == 'ecc_euclidean':
            return self.ecc_registration(img_fixed, img_moving, cv2.MOTION_EUCLIDEAN)
        elif method == 'ecc_translation':
            return self.ecc_registration(img_fixed, img_moving, cv2.MOTION_TRANSLATION)
        elif method == 'phase_correlation':
            return self.phase_correlation_registration(img_fixed, img_moving)
        elif method == 'feature_orb':
            return self.feature_registration(img_fixed, img_moving)
        else:
            raise ValueError(f"Unknown registration method: {method}")
    
    def ecc_registration(self, img_fixed, img_moving, motion_type):
        """Enhanced correlation coefficient registration
        
        Args:
            img_fixed: Fixed image
            img_moving: Moving image
            motion_type: cv2 motion model
            
        Returns:
            Tuple of (transform_matrix, quality)
        """
        try:
            img_fixed_proc = self.preprocess_image(img_fixed)
            img_moving_proc = self.preprocess_image(img_moving)
            
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-7)
            
            fixed_uint8 = (img_fixed_proc * 255).astype(np.uint8)
            moving_uint8 = (img_moving_proc * 255).astype(np.uint8)
            
            (cc, warp_matrix) = cv2.findTransformECC(
                fixed_uint8, moving_uint8, warp_matrix, motion_type, criteria
            )
            
            rows, cols = img_fixed.shape
            registered_img = cv2.warpAffine(
                img_moving, warp_matrix, (cols, rows),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )
            
            quality = self.calculate_multiscale_similarity(img_fixed, registered_img)
            return warp_matrix, quality
            
        except Exception as e:
            print(f"ECC registration error: {e}")
            return np.eye(2, 3, dtype=np.float32), 0.0
    
    def phase_correlation_registration(self, img_fixed, img_moving):
        """Phase correlation-based registration
        
        Args:
            img_fixed: Fixed image
            img_moving: Moving image
            
        Returns:
            Tuple of (transform_matrix, quality)
        """
        try:
            img_fixed_proc = self.preprocess_image(img_fixed)
            img_moving_proc = self.preprocess_image(img_moving)
            
            h, w = img_fixed_proc.shape
            img_moving_proc = cv2.resize(img_moving_proc, (w, h))
            
            # Hanning window
            window = np.outer(np.hanning(h), np.hanning(w))
            img_fixed_win = img_fixed_proc * window
            img_moving_win = img_moving_proc * window
            
            # FFT
            f_fixed = np.fft.fft2(img_fixed_win)
            f_moving = np.fft.fft2(img_moving_win)
            
            # Phase correlation
            cross_power_spectrum = (f_fixed * np.conj(f_moving)) / \
                                 (np.abs(f_fixed * np.conj(f_moving)) + 1e-10)
            correlation = np.fft.ifft2(cross_power_spectrum)
            
            # Find peak
            correlation_abs = np.abs(correlation)
            y_shift, x_shift = np.unravel_index(np.argmax(correlation_abs), correlation_abs.shape)
            
            # Handle periodicity
            if y_shift > h // 2:
                y_shift -= h
            if x_shift > w // 2:
                x_shift -= w
            
            # Limit shift
            max_shift = min(h, w) // 4
            x_shift = np.clip(x_shift, -max_shift, max_shift)
            y_shift = np.clip(y_shift, -max_shift, max_shift)
            
            transform = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
            registered_img = cv2.warpAffine(img_moving, transform, (w, h))
            quality = self.calculate_multiscale_similarity(img_fixed, registered_img)
            
            return transform, quality
            
        except Exception as e:
            print(f"Phase correlation error: {e}")
            return np.eye(2, 3, dtype=np.float32), 0.0
    
    def feature_registration(self, img_fixed, img_moving):
        """Feature-based registration using ORB/SIFT
        
        Args:
            img_fixed: Fixed image
            img_moving: Moving image
            
        Returns:
            Tuple of (transform_matrix, quality)
        """
        try:
            img_fixed_proc = self.preprocess_image(img_fixed)
            img_moving_proc = self.preprocess_image(img_moving)
            
            fixed_uint8 = (img_fixed_proc * 255).astype(np.uint8)
            moving_uint8 = (img_moving_proc * 255).astype(np.uint8)
            
            # Try SIFT, fall back to ORB
            try:
                detector = cv2.SIFT_create(nfeatures=1500)
                kp1, des1 = detector.detectAndCompute(fixed_uint8, None)
                kp2, des2 = detector.detectAndCompute(moving_uint8, None)
                use_sift = True
            except:
                detector = cv2.ORB_create(nfeatures=1500)
                kp1, des1 = detector.detectAndCompute(fixed_uint8, None)
                kp2, des2 = detector.detectAndCompute(moving_uint8, None)
                use_sift = False
            
            if des1 is None or des2 is None or len(des1) < 20 or len(des2) < 20:
                raise ValueError("Insufficient keypoints")
            
            # Feature matching
            if use_sift:
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                matches = matcher.knnMatch(des1, des2, k=2)
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
            else:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(des1, des2)
                good_matches = sorted(matches, key=lambda x: x.distance)[:100]
            
            if len(good_matches) < 15:
                raise ValueError(f"Insufficient matches: {len(good_matches)}")
            
            src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            transform, mask = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, 
                method=cv2.RANSAC, 
                ransacReprojThreshold=2.0,
                maxIters=2000,
                confidence=0.99
            )
            
            if transform is None:
                raise ValueError("Cannot estimate transform")
            
            inlier_ratio = np.sum(mask) / len(mask) if mask is not None else 0
            if inlier_ratio < 0.3:
                raise ValueError(f"Inlier ratio too low: {inlier_ratio:.2f}")
            
            rows, cols = img_fixed.shape
            registered_img = cv2.warpAffine(img_moving, transform, (cols, rows))
            quality = self.calculate_multiscale_similarity(img_fixed, registered_img)
            
            return transform, quality
            
        except Exception as e:
            print(f"Feature registration error: {e}")
            return np.eye(2, 3, dtype=np.float32), 0.0
    
    def load_image_stack(self, modality_dir, image_type='ori'):
        """Load image stack from directory
        
        Args:
            modality_dir: Path to modality directory
            image_type: 'ori' or 'mask'
            
        Returns:
            Tuple of (image_array, slice_indices)
        """
        image_dir = modality_dir / image_type
        if not image_dir.exists():
            return None, []
        
        image_files = sorted(list(image_dir.glob(f"{image_type}_*.png")))
        if not image_files:
            return None, []
        
        images = []
        slice_indices = []
        
        for img_file in image_files:
            try:
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img.astype(np.float32) / 255.0)
                    slice_idx = int(img_file.stem.split('_')[-1])
                    slice_indices.append(slice_idx)
            except Exception as e:
                print(f"Warning: Cannot read {img_file}: {e}")
                continue
        
        if images:
            return np.array(images), slice_indices
        return None, []
    
    def find_slice_correspondence_enhanced(self, stack1, stack2, indices1, indices2):
        """Enhanced slice correspondence finding
        
        Args:
            stack1: Reference image stack
            stack2: Moving image stack
            indices1: Slice indices for stack1
            indices2: Slice indices for stack2
            
        Returns:
            List of correspondence tuples
        """
        print(f"Enhanced matching: {len(stack1)} vs {len(stack2)} slices")
        
        correspondence = []
        
        # Check layer count ratio
        ratio_diff = abs(len(stack1) - len(stack2)) / max(len(stack1), len(stack2))
        
        if ratio_diff <= 0.3:
            correspondence = self.sequential_matching(stack1, stack2, indices1, indices2)
        else:
            correspondence = self.keypoint_matching(stack1, stack2, indices1, indices2)
        
        # Quality filtering and optimization
        correspondence = self.optimize_correspondence(correspondence, stack1, stack2)
        
        print(f"Enhanced matching found {len(correspondence)} pairs")
        return correspondence
    
    def sequential_matching(self, stack1, stack2, indices1, indices2):
        """Sequential matching strategy
        
        Args:
            stack1, stack2: Image stacks
            indices1, indices2: Slice indices
            
        Returns:
            List of correspondence tuples
        """
        correspondence = []
        ratio = len(stack2) / len(stack1)
        
        for i, img1 in enumerate(stack1):
            estimated_j = int(i * ratio)
            search_range = max(2, min(5, len(stack2) // 15))
            start_j = max(0, estimated_j - search_range)
            end_j = min(len(stack2), estimated_j + search_range + 1)
            
            best_sim = 0
            best_j = -1
            
            for j in range(start_j, end_j):
                sim = self.calculate_multiscale_similarity(img1, stack2[j])
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            
            if best_j != -1 and best_sim > self.similarity_threshold:
                correspondence.append((i, best_j, best_sim, indices1[i], indices2[best_j]))
        
        return correspondence
    
    def keypoint_matching(self, stack1, stack2, indices1, indices2):
        """Keypoint matching strategy for large layer differences
        
        Args:
            stack1, stack2: Image stacks
            indices1, indices2: Slice indices
            
        Returns:
            List of correspondence tuples
        """
        correspondence = []
        n_keypoints = min(10, min(len(stack1), len(stack2)) // 3)
        
        if n_keypoints < 3:
            return self.global_matching(stack1, stack2, indices1, indices2)
        
        key_indices1 = np.linspace(0, len(stack1)-1, n_keypoints, dtype=int)
        key_indices2 = np.linspace(0, len(stack2)-1, n_keypoints, dtype=int)
        
        # Find keypoint correspondences
        key_matches = []
        for i in key_indices1:
            best_sim = 0
            best_j = -1
            
            for j in key_indices2:
                sim = self.calculate_multiscale_similarity(stack1[i], stack2[j])
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            
            if best_sim > self.similarity_threshold:
                key_matches.append((i, best_j, best_sim))
        
        if len(key_matches) < 3:
            return self.global_matching(stack1, stack2, indices1, indices2)
        
        # Interpolate other slices based on keypoints
        for i in range(len(stack1)):
            j_interp = self.interpolate_match_position(i, key_matches, len(stack2))
            
            if 0 <= j_interp < len(stack2):
                sim = self.calculate_multiscale_similarity(stack1[i], stack2[j_interp])
                if sim > self.similarity_threshold:
                    correspondence.append((i, j_interp, sim, indices1[i], indices2[j_interp]))
        
        return correspondence
    
    def global_matching(self, stack1, stack2, indices1, indices2):
        """Global matching strategy
        
        Args:
            stack1, stack2: Image stacks
            indices1, indices2: Slice indices
            
        Returns:
            List of correspondence tuples
        """
        correspondence = []
        used_j = set()
        
        # Create similarity matrix
        similarity_matrix = np.zeros((len(stack1), len(stack2)))
        
        for i, img1 in enumerate(stack1):
            for j, img2 in enumerate(stack2):
                similarity_matrix[i, j] = self.calculate_multiscale_similarity(img1, img2)
        
        # Greedy matching
        for _ in range(min(len(stack1), len(stack2))):
            max_pos = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
            i, j = max_pos
            max_sim = similarity_matrix[i, j]
            
            if max_sim > self.similarity_threshold and j not in used_j:
                correspondence.append((i, j, max_sim, indices1[i], indices2[j]))
                used_j.add(j)
                
                similarity_matrix[i, :] = 0
                similarity_matrix[:, j] = 0
            else:
                break
        
        return correspondence
    
    def optimize_correspondence(self, correspondence, stack1, stack2):
        """Optimize correspondences by checking continuity
        
        Args:
            correspondence: List of correspondence tuples
            stack1: Reference stack
            stack2: Moving stack
            
        Returns:
            Optimized correspondence list
        """
        if len(correspondence) < 3:
            return correspondence
        
        correspondence.sort(key=lambda x: x[0])
        
        optimized = []
        prev_ratio = None
        
        for i, corr in enumerate(correspondence):
            i1, i2, sim, idx1, idx2 = corr
            
            if i == 0:
                optimized.append(corr)
                prev_ratio = i2 / max(1, i1)
            else:
                current_ratio = i2 / max(1, i1)
                
                if prev_ratio is not None and abs(current_ratio - prev_ratio) > 0.5:
                    expected_i2 = int(i1 * prev_ratio)
                    search_range = 3
                    
                    best_sim = sim
                    best_i2 = i2
                    best_idx2 = idx2
                    
                    for test_i2 in range(max(0, expected_i2 - search_range),
                                       min(len(stack2), expected_i2 + search_range + 1)):
                        if test_i2 < len(stack2):
                            test_sim = self.calculate_multiscale_similarity(
                                stack1[i1], stack2[test_i2]
                            )
                            if test_sim > best_sim * 0.9:
                                best_sim = test_sim
                                best_i2 = test_i2
                    
                    optimized.append((i1, best_i2, best_sim, idx1, best_idx2))
                    prev_ratio = best_i2 / max(1, i1)
                else:
                    optimized.append(corr)
                    prev_ratio = current_ratio
        
        return optimized
    
    def interpolate_match_position(self, i, key_matches, max_j):
        """Interpolate match position
        
        Args:
            i: Index to interpolate
            key_matches: Keypoint matches
            max_j: Maximum j value
            
        Returns:
            Interpolated j position
        """
        if not key_matches:
            return 0
        
        key_matches_sorted = sorted(key_matches, key=lambda x: x[0])
        
        lower_match = None
        upper_match = None
        
        for match in key_matches_sorted:
            if match[0] <= i:
                lower_match = match
            if match[0] >= i and upper_match is None:
                upper_match = match
                break
        
        if lower_match and upper_match and lower_match[0] != upper_match[0]:
            ratio = (i - lower_match[0]) / (upper_match[0] - lower_match[0])
            j_interp = int(lower_match[1] + ratio * (upper_match[1] - lower_match[1]))
            return max(0, min(max_j - 1, j_interp))
        elif lower_match:
            return lower_match[1]
        elif upper_match:
            return upper_match[1]
        else:
            ratio = len(key_matches_sorted) / max_j if key_matches_sorted else 1
            return min(max_j - 1, int(i / ratio))
    
    def advanced_two_stage_registration(self, ref_data, moving_data, correspondence):
        """Advanced two-stage registration
        
        Args:
            ref_data: Reference modality data
            moving_data: Moving modality data
            correspondence: Slice correspondences
            
        Returns:
            Dictionary with registered data
        """
        print("Executing advanced two-stage registration...")
        
        # Stage 1: Global alignment
        global_transform = self.estimate_global_transform(
            ref_data['ori_stack'], moving_data['ori_stack'],
            ref_data['ori_indices'], moving_data['ori_indices']
        )
        
        # Stage 2: Slice-by-slice fine alignment
        registered_ori_stack = []
        registered_mask_stack = []
        transform_matrices = []
        target_indices = ref_data['ori_indices']
        
        # Create correspondence dictionary
        corr_dict = {}
        for corr in correspondence:
            ref_idx, mov_idx = corr[3], corr[4]
            corr_dict[ref_idx] = corr[1]
        
        successful_registrations = 0
        total_quality = 0
        quality_scores = []
        
        print("Fine registration by slice...")
        for ref_i, ref_idx in enumerate(tqdm(target_indices, desc="Registration", leave=False)):
            ref_img = ref_data['ori_stack'][ref_i]
            
            if ref_idx in corr_dict:
                mov_i = corr_dict[ref_idx]
                if mov_i < len(moving_data['ori_stack']):
                    mov_img = moving_data['ori_stack'][mov_i]
                    
                    rows, cols = ref_img.shape
                    globally_aligned = cv2.warpAffine(
                        mov_img, global_transform, (cols, rows),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                    )
                    
                    fine_transform, quality = self.multi_method_registration(
                        ref_img, globally_aligned
                    )
                    
                    combined_transform = self.combine_transforms(global_transform, fine_transform)
                    
                    final_registered = cv2.warpAffine(
                        mov_img, combined_transform, (cols, rows),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                    )
                    
                    final_quality = self.calculate_multiscale_similarity(ref_img, final_registered)
                    
                    if final_quality > self.min_registration_quality:
                        registered_ori_stack.append(final_registered)
                        transform_matrices.append(combined_transform)
                        quality_scores.append(final_quality)
                        successful_registrations += 1
                        total_quality += final_quality
                        
                        # Process mask
                        if (moving_data['mask_stack'] is not None and 
                            mov_i < len(moving_data['mask_stack'])):
                            mov_mask = moving_data['mask_stack'][mov_i]
                            registered_mask = cv2.warpAffine(
                                mov_mask, combined_transform, (cols, rows),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
                            )
                            registered_mask_stack.append(registered_mask)
                        else:
                            registered_mask_stack.append(np.zeros_like(ref_img))
                    else:
                        registered_ori_stack.append(globally_aligned)
                        registered_mask_stack.append(np.zeros_like(ref_img))
                        transform_matrices.append(global_transform)
                        quality_scores.append(self.calculate_multiscale_similarity(ref_img, globally_aligned))
                else:
                    registered_ori_stack.append(np.zeros_like(ref_img))
                    registered_mask_stack.append(np.zeros_like(ref_img))
                    transform_matrices.append(np.eye(2, 3, dtype=np.float32))
                    quality_scores.append(0.0)
            else:
                registered_ori_stack.append(np.zeros_like(ref_img))
                registered_mask_stack.append(np.zeros_like(ref_img))
                transform_matrices.append(np.eye(2, 3, dtype=np.float32))
                quality_scores.append(0.0)
        
        success_rate = successful_registrations / len(target_indices) if target_indices else 0
        avg_quality = total_quality / successful_registrations if successful_registrations > 0 else 0
        
        print(f"Two-stage registration complete: {successful_registrations}/{len(target_indices)} successful")
        print(f"Average quality: {avg_quality:.3f}, Std: {np.std(quality_scores):.3f}")
        
        return {
            'ori_stack': np.array(registered_ori_stack),
            'ori_indices': target_indices,
            'mask_stack': np.array(registered_mask_stack) if registered_mask_stack else None,
            'mask_indices': target_indices,
            'transforms': transform_matrices,
            'success_rate': success_rate,
            'avg_quality': avg_quality,
            'quality_scores': quality_scores,
            'global_transform': global_transform
        }
    
    def estimate_global_transform(self, ref_stack, moving_stack, indices1, indices2):
        """Estimate global transformation
        
        Args:
            ref_stack: Reference image stack
            moving_stack: Moving image stack
            indices1: Reference slice indices
            indices2: Moving slice indices
            
        Returns:
            Global transformation matrix
        """
        print("Estimating global transformation...")
        
        n_samples = min(self.global_registration_samples, 
                       min(len(ref_stack), len(moving_stack)))
        
        ref_indices = np.linspace(0, len(ref_stack)-1, n_samples, dtype=int)
        moving_indices = np.linspace(0, len(moving_stack)-1, n_samples, dtype=int)
        
        transforms = []
        qualities = []
        
        print(f"Using {n_samples} samples for global estimation")
        
        for ref_i, mov_i in zip(ref_indices, moving_indices):
            ref_img = ref_stack[ref_i]
            mov_img = moving_stack[mov_i]
            
            transform, quality = self.multi_method_registration(ref_img, mov_img)
            
            if quality > self.min_registration_quality:
                transforms.append(transform)
                qualities.append(quality)
        
        if not transforms:
            print("Global transform estimation failed, using identity")
            return np.eye(2, 3, dtype=np.float32)
        
        # Filter outliers
        if len(transforms) > 3:
            transforms, qualities = self.filter_transform_outliers(transforms, qualities)
        
        # Weighted average
        qualities = np.array(qualities)
        weights = qualities / np.sum(qualities)
        
        avg_transform = np.zeros((2, 3), dtype=np.float32)
        for transform, weight in zip(transforms, weights):
            avg_transform += weight * transform
        
        print(f"Global transform estimated (based on {len(transforms)} valid samples)")
        return avg_transform
    
    def filter_transform_outliers(self, transforms, qualities):
        """Filter transform outliers using IQR method
        
        Args:
            transforms: List of transformation matrices
            qualities: List of quality scores
            
        Returns:
            Tuple of (filtered_transforms, filtered_qualities)
        """
        if len(transforms) <= 3:
            return transforms, qualities
        
        translations = np.array([t[:, 2] for t in transforms])
        translation_distances = np.linalg.norm(translations, axis=1)
        
        q1 = np.percentile(translation_distances, 25)
        q3 = np.percentile(translation_distances, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        valid_indices = []
        for i, dist in enumerate(translation_distances):
            if lower_bound <= dist <= upper_bound:
                valid_indices.append(i)
        
        if len(valid_indices) < len(transforms) // 2:
            sorted_indices = np.argsort(qualities)[::-1]
            valid_indices = sorted_indices[:len(transforms)//2]
        
        filtered_transforms = [transforms[i] for i in valid_indices]
        filtered_qualities = [qualities[i] for i in valid_indices]
        
        print(f"Filtered outliers: {len(transforms)} -> {len(filtered_transforms)}")
        return filtered_transforms, filtered_qualities
    
    def combine_transforms(self, global_transform, local_transform):
        """Combine global and local transforms
        
        Args:
            global_transform: Global transformation
            local_transform: Local refinement
            
        Returns:
            Combined transformation matrix
        """
        global_3x3 = np.vstack([global_transform, [0, 0, 1]])
        local_3x3 = np.vstack([local_transform, [0, 0, 1]])
        
        combined_3x3 = local_3x3 @ global_3x3
        
        return combined_3x3[:2, :].astype(np.float32)
    
    def process_patient_registration(self, hospital_name, patient_id):
        """Process registration for a single patient
        
        Args:
            hospital_name: Hospital identifier
            patient_id: Patient identifier
            
        Returns:
            Tuple of (success_flag, message)
        """
        source_patient_dir = self.source_root / hospital_name / patient_id
        target_patient_dir = self.target_root / hospital_name / patient_id
        
        if not source_patient_dir.exists():
            return False, "Patient directory not found"
        
        print(f"Processing patient: {patient_id}")
        
        # Check available modalities
        modalities = ['T1', 'T2', 'T1C']
        available_modalities = {}
        
        for modality in modalities:
            modality_dir = source_patient_dir / modality
            if modality_dir.exists():
                ori_stack, ori_indices = self.load_image_stack(modality_dir, 'ori')
                mask_stack, mask_indices = self.load_image_stack(modality_dir, 'mask')
                
                if ori_stack is not None and len(ori_stack) > 3:
                    available_modalities[modality] = {
                        'ori_stack': ori_stack,
                        'ori_indices': ori_indices,
                        'mask_stack': mask_stack,
                        'mask_indices': mask_indices,
                        'dir': modality_dir
                    }
        
        if len(available_modalities) < 1:
            return False, "No available modalities"
        
        print(f"Found modalities: {list(available_modalities.keys())}")
        
        # Check layer differences
        layer_counts = {mod: len(data['ori_stack']) for mod, data in available_modalities.items()}
        max_layers = max(layer_counts.values())
        min_layers = min(layer_counts.values())
        
        print(f"Layer counts: {layer_counts}")
        
        if min_layers / max_layers < (1 - self.max_layer_diff_ratio):
            return False, f"Layer difference too large: {min_layers}/{max_layers}"
        
        # Handle single modality case
        if len(available_modalities) == 1:
            print("Single modality detected, saving directly")
            modality = list(available_modalities.keys())[0]
            print(f"Modality: {modality} ({layer_counts[modality]} slices)")
            
            registration_results = {modality: available_modalities[modality]}
            registration_stats = {
                'total_attempts': 0,
                'successful_registrations': 1,
                'quality_scores': [1.0]
            }
            
            self.save_registration_results(registration_results, target_patient_dir, registration_stats)
            return True, f"Single modality {modality} saved ({layer_counts[modality]} slices)"
        
        # Select reference modality
        if 'T1' in available_modalities:
            reference_modality = 'T1'
        else:
            reference_modality = max(available_modalities.keys(), 
                                   key=lambda x: len(available_modalities[x]['ori_stack']))
        
        print(f"Reference modality: {reference_modality} ({layer_counts[reference_modality]} slices)")
        
        # Register other modalities
        registration_results = {}
        registration_results[reference_modality] = available_modalities[reference_modality]
        
        ref_data = available_modalities[reference_modality]
        successful_registrations = 1
        registration_stats = {
            'total_attempts': 0,
            'successful_registrations': 1,
            'quality_scores': []
        }
        
        for modality, data in available_modalities.items():
            if modality == reference_modality:
                continue
            
            print(f"Registering {modality} -> {reference_modality}")
            registration_stats['total_attempts'] += 1
            
            try:
                # Enhanced matching
                correspondence = self.find_slice_correspondence_enhanced(
                    ref_data['ori_stack'], data['ori_stack'],
                    ref_data['ori_indices'], data['ori_indices']
                )
                
                min_required = max(3, min(len(ref_data['ori_stack']), len(data['ori_stack'])) * self.min_correspondence_ratio)
                
                if len(correspondence) < min_required:
                    print(f"Skipped: Too few corresponding slices ({len(correspondence)} < {min_required:.0f})")
                    continue
                
                # Advanced two-stage registration
                registered_data = self.advanced_two_stage_registration(ref_data, data, correspondence)
                
                if registered_data and registered_data['success_rate'] > 0.15:
                    registration_results[modality] = registered_data
                    successful_registrations += 1
                    registration_stats['successful_registrations'] += 1
                    registration_stats['quality_scores'].extend(registered_data.get('quality_scores', []))
                    
                    print(f"✓ {modality} registration successful")
                    print(f"  Success rate: {registered_data['success_rate']:.2f}")
                    print(f"  Average quality: {registered_data['avg_quality']:.3f}")
                    
                    if 'quality_scores' in registered_data:
                        quality_std = np.std(registered_data['quality_scores'])
                        print(f"  Quality std: {quality_std:.3f}")
                else:
                    print(f"✗ {modality} registration failed")
                
            except Exception as e:
                print(f"✗ {modality} registration error: {e}")
                continue
        
        # Save results
        if successful_registrations >= 1:
            self.save_registration_results(registration_results, target_patient_dir, registration_stats)
            
            avg_quality = np.mean(registration_stats['quality_scores']) if registration_stats['quality_scores'] else 0
            if successful_registrations == 1:
                return True, f"Saved {successful_registrations} modality (reference)"
            else:
                return True, f"Successfully registered {successful_registrations} modalities (avg quality: {avg_quality:.3f})"
        else:
            return False, "No valid modality data"
    
    def process_hospital(self, hospital_name):
        """Process registration for a single hospital
        
        Args:
            hospital_name: Hospital identifier
            
        Returns:
            Tuple of (successful_count, failed_count, failed_patients_list)
        """
        hospital_dir = self.source_root / hospital_name
        if not hospital_dir.exists():
            return 0, 0, []
        
        print(f"\n{'='*60}")
        print(f"Processing hospital: {hospital_name}")
        print(f"{'='*60}")
        
        patient_dirs = [d for d in hospital_dir.iterdir() if d.is_dir()]
        patient_dirs.sort()
        
        successful = 0
        failed = 0
        failed_patients = []
        
        for patient_dir in tqdm(patient_dirs, desc=f"Registration-{hospital_name}"):
            try:
                success, message = self.process_patient_registration(hospital_name, patient_dir.name)
                
                if success:
                    successful += 1
                    print(f"✓ {message}")
                else:
                    failed += 1
                    failed_patients.append((patient_dir.name, message))
                    print(f"✗ {message}")
                    
            except Exception as e:
                failed += 1
                error_msg = f"Processing error: {str(e)}"
                failed_patients.append((patient_dir.name, error_msg))
                print(f"✗ {error_msg}")
        
        print(f"\n{hospital_name} completed: {successful} successful, {failed} failed")
        return successful, failed, failed_patients
    
    def register_all_data(self):
        """Register all data across all hospitals
        
        Returns:
            Dictionary of failed patients by hospital
        """
        hospitals = ['960', 'BJ', 'HX', 'QD', 'QL', 'TJ']
        
        print("Starting advanced multi-modal registration")
        print(f"Source directory: {self.source_root}")
        print(f"Target directory: {self.target_root}")
        print("\nRegistration parameters:")
        print(f"  Similarity threshold: {self.similarity_threshold}")
        print(f"  Max layer diff ratio: {self.max_layer_diff_ratio}")
        print(f"  Min correspondence ratio: {self.min_correspondence_ratio}")
        print(f"  Global samples: {self.global_registration_samples}")
        print(f"  Multi-scale: {self.use_multiscale}")
        print(f"  Quality threshold: {self.quality_threshold}")
        
        total_successful = 0
        total_failed = 0
        all_failed_patients = {}
        
        for hospital in hospitals:
            try:
                successful, failed, failed_patients = self.process_hospital(hospital)
                total_successful += successful
                total_failed += failed
                
                if failed_patients:
                    all_failed_patients[hospital] = failed_patients
                
            except Exception as e:
                print(f"Error processing hospital {hospital}: {e}")
                continue
        
        # Generate report
        self.generate_comprehensive_report(total_successful, total_failed, all_failed_patients)
        
        return all_failed_patients
    
    def generate_comprehensive_report(self, total_successful, total_failed, failed_patients):
        """Generate comprehensive registration report
        
        Args:
            total_successful: Count of successful registrations
            total_failed: Count of failed registrations
            failed_patients: Dictionary of failed patients
        """
        print(f"\n{'='*60}")
        print("🏥 Advanced Multi-Modal Registration Summary Report")
        print(f"{'='*60}")
        
        total_patients = total_successful + total_failed
        if total_patients > 0:
            success_rate = total_successful / total_patients * 100
            print(f"📊 Overall Statistics:")
            print(f"  Total patients processed: {total_patients}")
            print(f"  Successful registrations: {total_successful} ({success_rate:.1f}%)")
            print(f"  Failed registrations: {total_failed} ({100-success_rate:.1f}%)")
        
        if failed_patients:
            print(f"\n❌ Failure Details:")
            
            all_failure_reasons = {}
            for hospital, patients in failed_patients.items():
                print(f"\n  Hospital {hospital}:")
                hospital_reasons = {}
                
                for patient_id, reason in patients:
                    main_reason = reason.split(':')[0].split('(')[0].strip()
                    
                    if main_reason not in hospital_reasons:
                        hospital_reasons[main_reason] = []
                    hospital_reasons[main_reason].append(patient_id)
                    
                    if main_reason not in all_failure_reasons:
                        all_failure_reasons[main_reason] = 0
                    all_failure_reasons[main_reason] += 1
                
                for reason, patient_list in hospital_reasons.items():
                    print(f"    {reason}: {len(patient_list)} patients")
            
            print(f"\n  Global Failure Reasons:")
            sorted_reasons = sorted(all_failure_reasons.items(), key=lambda x: x[1], reverse=True)
            for reason, count in sorted_reasons:
                percentage = count / total_failed * 100
                print(f"    {reason}: {count} cases ({percentage:.1f}%)")
        
        print(f"\n💡 Improvement Suggestions:")
        print("  • similarity_threshold: current {}, range 0.04-0.10".format(self.similarity_threshold))
        print("  • min_correspondence_ratio: current {}, range 0.05-0.20".format(self.min_correspondence_ratio))
        print("  • quality_threshold: current {}, range 0.10-0.25".format(self.quality_threshold))
        
        print(f"\n📁 Registration results saved to: {self.target_root}")
        print("   Each patient directory contains:")
        print("   • ori/ - registered images")
        print("   • mask/ - registered masks")
        print("   • registration_info.json - detailed registration info")
        print("   • registration_summary.json - registration statistics")
        
        print(f"\n{'='*60}")
        print("🎉 Advanced multi-modal registration completed!")
        print(f"{'='*60}")


def main():
    """Main entry point"""
    print("🔄 Advanced Multi-Modal MRI Registration Tool (V2)")
    print("="*70)
    
    # Configuration
    source_root = "./data/png_slices"
    target_root = "./data/registered"
    
    print(f"📂 Source directory: {source_root}")
    print(f"📁 Target directory: {target_root}\n")
    
    # Create registrator
    registration = AdvancedMultiModalRegistration(source_root, target_root)
    
    print("📋 Parameter Suggestions:")
    print("   For lower success rate, try:")
    print("   • Lower similarity_threshold (e.g., 0.04)")
    print("   • Lower min_correspondence_ratio (e.g., 0.05)")
    print("   • Lower quality_threshold (e.g., 0.10)")
    print("   • Increase max_layer_diff_ratio (e.g., 0.95)\n")
    
    try:
        # Execute registration
        failed_patients = registration.register_all_data()
        
    except KeyboardInterrupt:
        print("\n⚠️  Registration interrupted by user")
    except Exception as e:
        print(f"❌ Critical error during registration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
