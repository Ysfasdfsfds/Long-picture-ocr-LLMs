#!/usr/bin/env python3
import json
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class OCRDifference:
    slice_index: int
    original_text: str
    current_text: str
    original_score: float
    current_score: float
    box: List[List[float]]


@dataclass
class AvatarDifference:
    slice_index: int
    original_box: List[int]
    current_box: List[int]
    original_center: Tuple[float, float]
    current_center: Tuple[float, float]
    distance: float


class JSONComparer:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.ocr_original_path = os.path.join(base_dir, 'all_ocr_results_original.json')
        self.ocr_current_path = os.path.join(base_dir, 'all_ocr_results.json')
        self.avatar_original_path = os.path.join(base_dir, 'all_avatar_positions_original.json')
        self.avatar_current_path = os.path.join(base_dir, 'all_avatar_positions.json')
        
    def load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Load JSON file and return data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_distance(self, center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two centers"""
        return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
    
    def compare_ocr_results(self) -> Dict[str, Any]:
        """Compare OCR results between original and current"""
        original_data = self.load_json(self.ocr_original_path)
        current_data = self.load_json(self.ocr_current_path)
        
        # Group by slice_index
        original_by_slice = defaultdict(list)
        current_by_slice = defaultdict(list)
        
        for item in original_data:
            original_by_slice[item['slice_index']].append(item)
        
        for item in current_data:
            current_by_slice[item['slice_index']].append(item)
        
        differences = []
        stats = {
            'total_original': len(original_data),
            'total_current': len(current_data),
            'text_changes': 0,
            'score_changes': 0,
            'new_detections': 0,
            'missing_detections': 0
        }
        
        # Compare items by matching boxes
        for slice_idx in set(list(original_by_slice.keys()) + list(current_by_slice.keys())):
            orig_items = original_by_slice.get(slice_idx, [])
            curr_items = current_by_slice.get(slice_idx, [])
            
            # Match items by box coordinates
            matched_current = set()
            
            for orig_item in orig_items:
                found_match = False
                for i, curr_item in enumerate(curr_items):
                    if i in matched_current:
                        continue
                    
                    # Check if boxes are similar (allowing small variations)
                    if self._boxes_match(orig_item['box'], curr_item['box']):
                        matched_current.add(i)
                        found_match = True
                        
                        # Compare text and score
                        if orig_item['text'] != curr_item['text']:
                            stats['text_changes'] += 1
                            differences.append(OCRDifference(
                                slice_index=slice_idx,
                                original_text=orig_item['text'],
                                current_text=curr_item['text'],
                                original_score=orig_item['score'],
                                current_score=curr_item['score'],
                                box=orig_item['box']
                            ))
                        elif abs(orig_item['score'] - curr_item['score']) > 0.01:
                            stats['score_changes'] += 1
                        break
                
                if not found_match:
                    stats['missing_detections'] += 1
            
            # Check for new detections
            for i, curr_item in enumerate(curr_items):
                if i not in matched_current:
                    stats['new_detections'] += 1
        
        return {
            'differences': differences,
            'statistics': stats
        }
    
    def _boxes_match(self, box1: List[List[float]], box2: List[List[float]], tolerance: float = 5.0) -> bool:
        """Check if two boxes match within tolerance"""
        if len(box1) != len(box2):
            return False
        
        for p1, p2 in zip(box1, box2):
            if abs(p1[0] - p2[0]) > tolerance or abs(p1[1] - p2[1]) > tolerance:
                return False
        return True
    
    def compare_avatar_positions(self) -> Dict[str, Any]:
        """Compare avatar positions between original and current"""
        original_data = self.load_json(self.avatar_original_path)
        current_data = self.load_json(self.avatar_current_path)
        
        # Group by slice_index
        original_by_slice = defaultdict(list)
        current_by_slice = defaultdict(list)
        
        for item in original_data:
            original_by_slice[item['slice_index']].append(item)
        
        for item in current_data:
            current_by_slice[item['slice_index']].append(item)
        
        differences = []
        stats = {
            'total_original': len(original_data),
            'total_current': len(current_data),
            'position_changes': 0,
            'new_avatars': 0,
            'missing_avatars': 0,
            'avg_distance': 0.0
        }
        
        total_distance = 0.0
        distance_count = 0
        
        # Compare avatars by matching centers
        for slice_idx in set(list(original_by_slice.keys()) + list(current_by_slice.keys())):
            orig_items = original_by_slice.get(slice_idx, [])
            curr_items = current_by_slice.get(slice_idx, [])
            
            matched_current = set()
            
            for orig_item in orig_items:
                orig_center = (orig_item['center_x'], orig_item['center_y'])
                found_match = False
                min_distance = float('inf')
                best_match_idx = -1
                
                for i, curr_item in enumerate(curr_items):
                    if i in matched_current:
                        continue
                    
                    curr_center = (curr_item['center_x'], curr_item['center_y'])
                    distance = self.calculate_distance(orig_center, curr_center)
                    
                    # Find closest match within reasonable distance
                    if distance < min_distance and distance < 50:  # 50 pixel threshold
                        min_distance = distance
                        best_match_idx = i
                
                if best_match_idx >= 0:
                    matched_current.add(best_match_idx)
                    found_match = True
                    curr_item = curr_items[best_match_idx]
                    
                    if min_distance > 5:  # Consider it a position change if > 5 pixels
                        stats['position_changes'] += 1
                        differences.append(AvatarDifference(
                            slice_index=slice_idx,
                            original_box=orig_item['box'],
                            current_box=curr_item['box'],
                            original_center=orig_center,
                            current_center=(curr_item['center_x'], curr_item['center_y']),
                            distance=min_distance
                        ))
                    
                    total_distance += min_distance
                    distance_count += 1
                
                if not found_match:
                    stats['missing_avatars'] += 1
            
            # Check for new avatars
            for i, curr_item in enumerate(curr_items):
                if i not in matched_current:
                    stats['new_avatars'] += 1
        
        if distance_count > 0:
            stats['avg_distance'] = total_distance / distance_count
        
        return {
            'differences': differences,
            'statistics': stats
        }
    
    def generate_report(self, ocr_results: Dict[str, Any], avatar_results: Dict[str, Any]) -> str:
        """Generate a comprehensive comparison report"""
        report = []
        report.append("=" * 80)
        report.append("JSON Comparison Report")
        report.append("=" * 80)
        report.append("")
        
        # OCR Results
        report.append("## OCR Results Comparison")
        report.append("-" * 40)
        report.append(f"Total original items: {ocr_results['statistics']['total_original']}")
        report.append(f"Total current items: {ocr_results['statistics']['total_current']}")
        report.append(f"Text changes: {ocr_results['statistics']['text_changes']}")
        report.append(f"Score changes: {ocr_results['statistics']['score_changes']}")
        report.append(f"New detections: {ocr_results['statistics']['new_detections']}")
        report.append(f"Missing detections: {ocr_results['statistics']['missing_detections']}")
        report.append("")
        
        if ocr_results['differences']:
            report.append("### Text Changes Detail:")
            for diff in ocr_results['differences'][:10]:  # Show first 10
                report.append(f"  Slice {diff.slice_index}: '{diff.original_text}' -> '{diff.current_text}'")
                report.append(f"    Score: {diff.original_score:.4f} -> {diff.current_score:.4f}")
            if len(ocr_results['differences']) > 10:
                report.append(f"  ... and {len(ocr_results['differences']) - 10} more changes")
        report.append("")
        
        # Avatar Results
        report.append("## Avatar Positions Comparison")
        report.append("-" * 40)
        report.append(f"Total original avatars: {avatar_results['statistics']['total_original']}")
        report.append(f"Total current avatars: {avatar_results['statistics']['total_current']}")
        report.append(f"Position changes: {avatar_results['statistics']['position_changes']}")
        report.append(f"New avatars: {avatar_results['statistics']['new_avatars']}")
        report.append(f"Missing avatars: {avatar_results['statistics']['missing_avatars']}")
        report.append(f"Average position change: {avatar_results['statistics']['avg_distance']:.2f} pixels")
        report.append("")
        
        if avatar_results['differences']:
            report.append("### Position Changes Detail:")
            for diff in avatar_results['differences'][:10]:  # Show first 10
                report.append(f"  Slice {diff.slice_index}: moved {diff.distance:.2f} pixels")
                report.append(f"    From: ({diff.original_center[0]:.1f}, {diff.original_center[1]:.1f})")
                report.append(f"    To: ({diff.current_center[0]:.1f}, {diff.current_center[1]:.1f})")
            if len(avatar_results['differences']) > 10:
                report.append(f"  ... and {len(avatar_results['differences']) - 10} more changes")
        
        return "\n".join(report)
    
    def save_detailed_results(self, ocr_results: Dict[str, Any], avatar_results: Dict[str, Any]):
        """Save detailed comparison results to JSON"""
        detailed_results = {
            'ocr_comparison': {
                'statistics': ocr_results['statistics'],
                'differences': [
                    {
                        'slice_index': d.slice_index,
                        'original_text': d.original_text,
                        'current_text': d.current_text,
                        'original_score': d.original_score,
                        'current_score': d.current_score,
                        'box': d.box
                    } for d in ocr_results['differences']
                ]
            },
            'avatar_comparison': {
                'statistics': avatar_results['statistics'],
                'differences': [
                    {
                        'slice_index': d.slice_index,
                        'original_box': d.original_box,
                        'current_box': d.current_box,
                        'original_center': list(d.original_center),
                        'current_center': list(d.current_center),
                        'distance': d.distance
                    } for d in avatar_results['differences']
                ]
            }
        }
        
        output_path = os.path.join(os.path.dirname(__file__), 'comparison_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed results saved to: {output_path}")
    
    def run(self):
        """Run the comparison and generate reports"""
        print("Starting comparison...")
        
        # Compare OCR results
        print("Comparing OCR results...")
        ocr_results = self.compare_ocr_results()
        
        # Compare avatar positions
        print("Comparing avatar positions...")
        avatar_results = self.compare_avatar_positions()
        
        # Generate and print report
        report = self.generate_report(ocr_results, avatar_results)
        print("\n" + report)
        
        # Save detailed results
        self.save_detailed_results(ocr_results, avatar_results)
        
        # Save report to file
        report_path = os.path.join(os.path.dirname(__file__), 'comparison_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    comparer = JSONComparer()
    comparer.run()