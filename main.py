#!/usr/bin/env python3
"""
Execution script for quantum encoding of particle physics data.
"""

import os
import sys
import argparse
from quantum_encoding import encode_background_data


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Quantum encode particle physics data')
    parser.add_argument('--input', '-i', 
                       default='background_for_training.h5',
                       help='Path to input H5 file (default: background_for_training.h5)')
    parser.add_argument('--output', '-o',
                       default='encoded_particles_objectwise.npy',
                       help='Path to output file (default: encoded_particles_objectwise.npy)')
    parser.add_argument('--device', '-d',
                       default='lightning.gpu',
                       help='PennyLane device to use (default: lightning.gpu)')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress reporting during encoding')
    
    args = parser.parse_args()
    
    print("SLAC Research: Quantum Machine Learning for Particle Physics")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        print("Please check the file path or use --input to specify a different file.")
        return 1
    
    try:
        # Encode the background data
        encoded_data = encode_background_data(
            file_path=args.input,
            output_path=args.output,
            device_name=args.device,
            batch_progress=not args.no_progress
        )
        
        print("Quantum encoding completed successfully!")
        print(f"Encoded data shape: {encoded_data.shape}")
        print(f"Output saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Please check your file path and dependencies.")
        return 1


if __name__ == "__main__":
    sys.exit(main())