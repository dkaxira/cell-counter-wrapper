#!/usr/bin/env python3
"""
Batch Blood Cell Counter using HuggingFace Space API
Processes multiple blood smear images and saves results to CSV
"""

import os
import csv
import time
import shutil
from pathlib import Path
from gradio_client import Client, handle_file
import json
from datetime import datetime


class BloodCellBatchProcessor:
    def __init__(self, confidence_threshold=0.4):
        """
        Initialize the batch processor
        
        Args:
            confidence_threshold (float): Detection confidence threshold (0.0-1.0)
        """
        self.client = Client("dnth/blood-cell-counter")
        self.confidence_threshold = confidence_threshold
        self.results = []
        self.annotated_images_saved = 0
        
    def process_single_image(self, image_path, output_images_folder=None):
        """
        Process a single image and return results
        
        Args:
            image_path (str): Path to the image file
            output_images_folder (str): Path to save annotated images (optional)
            
        Returns:
            dict: Processing results
        """
        try:
            print(f"Processing: {os.path.basename(image_path)}")
            
            # Call the API
            result = self.client.predict(
                image=handle_file(image_path),
                confidence_threshold=self.confidence_threshold,
                api_name="/predict"
            )
            
            # Extract results: tuple of (detection_image_path, status_text, plot_data)
            detection_image_path, status_text, plot_data = result
            
            # Parse cell counts from plot_data
            cell_counts = {}
            total_cells = 0
            
            if plot_data and isinstance(plot_data, dict) and 'data' in plot_data:
                for row in plot_data['data']:
                    if len(row) >= 2:
                        cell_type = row[0]
                        count = row[1]
                        cell_counts[cell_type] = count
                        total_cells += count
                        
            # Extract summary from status text
            success_status = "successful" in status_text.lower()
            
            print(f"  ✅ Success! Total cells detected: {total_cells}")
            if cell_counts:
                for cell_type, count in cell_counts.items():
                    print(f"     • {cell_type}: {count}")
            
            # Save annotated image if requested and available
            saved_annotated_path = None
            if output_images_folder and detection_image_path and os.path.exists(detection_image_path):
                saved_annotated_path = self.save_annotated_image(
                    detection_image_path, 
                    image_path, 
                    output_images_folder
                )
                if saved_annotated_path:
                    print(f"  💾 Annotated image saved: {os.path.basename(saved_annotated_path)}")
            
            return {
                'filename': os.path.basename(image_path),
                'status': status_text,
                'success': success_status,
                'total_cells': total_cells,
                'cell_counts': cell_counts,
                'detection_image_path': detection_image_path,
                'saved_annotated_path': saved_annotated_path,
                'processed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            return {
                'filename': os.path.basename(image_path),
                'status': f'Error: {str(e)}',
                'success': False,
                'total_cells': 0,
                'cell_counts': {},
                'detection_image_path': None,
                'saved_annotated_path': None,
                'processed_at': datetime.now().isoformat()
            }
    
    def save_annotated_image(self, temp_image_path, original_image_path, output_folder):
        """
        Save the annotated image from temporary location to permanent folder
        
        Args:
            temp_image_path (str): Temporary path to annotated image from API
            original_image_path (str): Path to original image file
            output_folder (str): Folder to save annotated images
            
        Returns:
            str: Path to saved annotated image, or None if failed
        """
        try:
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Generate output filename
            original_name = Path(original_image_path).stem
            original_ext = Path(original_image_path).suffix
            
            # Keep original extension if possible, otherwise use .webp from API
            temp_ext = Path(temp_image_path).suffix if Path(temp_image_path).suffix else '.webp'
            if original_ext.lower() in ['.jpg', '.jpeg', '.png']:
                output_ext = original_ext
            else:
                output_ext = temp_ext
                
            output_filename = f"{original_name}_annotated{output_ext}"
            output_path = os.path.join(output_folder, output_filename)
            
            # Copy the annotated image
            shutil.copy2(temp_image_path, output_path)
            self.annotated_images_saved += 1
            
            return output_path
            
        except Exception as e:
            print(f"    ⚠️ Failed to save annotated image: {e}")
            return None
    
    def process_folder(self, input_folder, output_csv=None, output_images_folder=None, delay_seconds=1):
        """
        Process all images in a folder
        
        Args:
            input_folder (str): Path to folder containing blood smear images
            output_csv (str): Path to save results CSV (optional)
            output_images_folder (str): Path to save annotated images (optional)
            delay_seconds (float): Delay between API calls to avoid rate limiting
        """
        input_path = Path(input_folder)
        
        if not input_path.exists():
            raise ValueError(f"Input folder does not exist: {input_folder}")
        
        # Create output images folder if specified
        if output_images_folder:
            os.makedirs(output_images_folder, exist_ok=True)
            print(f"📁 Annotated images will be saved to: {output_images_folder}")
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No image files found in {input_folder}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] ", end="")
            
            result = self.process_single_image(str(image_file), output_images_folder)
            self.results.append(result)
            
            # Add delay to avoid overwhelming the API
            if i < len(image_files):  # Don't delay after the last image
                time.sleep(delay_seconds)
        
        # Save results
        if output_csv:
            self.save_results_to_csv(output_csv)
        
        print(f"\n✅ Completed processing {len(image_files)} images")
        if output_images_folder and self.annotated_images_saved > 0:
            print(f"💾 Saved {self.annotated_images_saved} annotated images to {output_images_folder}")
        
        return self.results
    
    def save_results_to_csv(self, output_path):
        """
        Save processing results to CSV file
        
        Args:
            output_path (str): Path to save the CSV file
        """
        if not self.results:
            print("No results to save")
            return
        
        # Prepare structured data for CSV
        processed_results = []
        
        # Get all unique cell types across all images
        all_cell_types = set()
        for result in self.results:
            if result.get('cell_counts'):
                all_cell_types.update(result['cell_counts'].keys())
        
        all_cell_types = sorted(all_cell_types)
        
        for result in self.results:
            row = {
                'filename': result['filename'],
                'success': result.get('success', False),
                'total_cells': result.get('total_cells', 0),
                'processed_at': result['processed_at']
            }
            
            # Add individual cell type counts
            cell_counts = result.get('cell_counts', {})
            for cell_type in all_cell_types:
                row[f'{cell_type}_count'] = cell_counts.get(cell_type, 0)
            
            # Add annotated image path if available
            if result.get('saved_annotated_path'):
                row['annotated_image_path'] = result['saved_annotated_path']
            
            # Add status (cleaned up)
            status = result['status'].replace('\n', ' | ').replace('\r', '')
            row['status'] = status
            
            processed_results.append(row)
        
        # Write to CSV
        if processed_results:
            fieldnames = list(processed_results[0].keys())
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(processed_results)
            
            print(f"📊 Results saved to: {output_path}")
    
    def print_summary(self):
        """Print a summary of processing results"""
        if not self.results:
            print("No results to summarize")
            return
        
        successful = len([r for r in self.results if r.get('success', False)])
        failed = len(self.results) - successful
        
        # Calculate total cells across all successful images
        total_cells_all = sum(r.get('total_cells', 0) for r in self.results if r.get('success', False))
        
        # Get cell type breakdown
        cell_type_totals = {}
        for result in self.results:
            if result.get('success', False) and result.get('cell_counts'):
                for cell_type, count in result['cell_counts'].items():
                    cell_type_totals[cell_type] = cell_type_totals.get(cell_type, 0) + count
        
        print(f"\n📋 Processing Summary:")
        print(f"  • Total images: {len(self.results)}")
        print(f"  • Successful: {successful}")
        print(f"  • Failed: {failed}")
        print(f"  • Total cells detected: {total_cells_all}")
        
        if cell_type_totals:
            print(f"\n🔬 Cell Type Breakdown:")
            for cell_type, count in sorted(cell_type_totals.items()):
                print(f"  • {cell_type}: {count}")
        
        if successful > 0:
            avg_cells = total_cells_all / successful
            print(f"\n📊 Statistics:")
            print(f"  • Average cells per image: {avg_cells:.1f}")
            
            # Per cell type averages
            if cell_type_totals:
                for cell_type, count in sorted(cell_type_totals.items()):
                    avg_per_type = count / successful
                    print(f"  • Average {cell_type} per image: {avg_per_type:.1f}")
        
        if failed > 0:
            print(f"\n❌ Failed files:")
            for result in self.results:
                if not result.get('success', False):
                    print(f"  • {result['filename']}: {result['status']}")
    
    def export_detailed_report(self, output_path):
        """
        Export a detailed JSON report with all data
        
        Args:
            output_path (str): Path to save the JSON report
        """
        report = {
            'summary': {
                'total_images': len(self.results),
                'successful': len([r for r in self.results if r.get('success', False)]),
                'failed': len([r for r in self.results if not r.get('success', False)]),
                'total_cells': sum(r.get('total_cells', 0) for r in self.results if r.get('success', False)),
                'confidence_threshold': self.confidence_threshold,
                'generated_at': datetime.now().isoformat()
            },
            'results': self.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Detailed report saved to: {output_path}")


def main():
    """
    Main function - example usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process blood cell images')
    parser.add_argument('input_folder', help='Path to folder containing blood smear images')
    parser.add_argument('-o', '--output', help='Output CSV file path', 
                       default='blood_cell_results.csv')
    parser.add_argument('-i', '--images', help='Output folder for annotated images (optional)')
    parser.add_argument('-j', '--json-report', help='Also save detailed JSON report',
                       action='store_true')
    parser.add_argument('-c', '--confidence', type=float, default=0.4,
                       help='Confidence threshold (0.0-1.0, default: 0.4)')
    parser.add_argument('-d', '--delay', type=float, default=1.0,
                       help='Delay between API calls in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    # Create processor
    processor = BloodCellBatchProcessor(confidence_threshold=args.confidence)
    
    try:
        # Process all images
        results = processor.process_folder(
            input_folder=args.input_folder,
            output_csv=args.output,
            output_images_folder=args.images,
            delay_seconds=args.delay
        )
        
        # Save detailed JSON report if requested
        if args.json_report:
            json_path = args.output.replace('.csv', '_detailed_report.json')
            processor.export_detailed_report(json_path)
        
        # Print summary
        processor.print_summary()
        
    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted by user")
        if processor.results:
            processor.save_results_to_csv(args.output)
            if args.json_report:
                json_path = args.output.replace('.csv', '_detailed_report.json')
                processor.export_detailed_report(json_path)
            processor.print_summary()
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Example usage if run directly
    print("Blood Cell Counter - Batch Processor")
    print("="*50)
    
    main()
