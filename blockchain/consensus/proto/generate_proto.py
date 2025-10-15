#!/usr/bin/env python3
"""
Script to generate gRPC Python files from .proto definitions
"""
import os
import subprocess
import sys

def generate_proto_files():
    """Generate gRPC Python files from .proto definition"""
    
    # Get the directory of this script
    proto_dir = os.path.dirname(os.path.abspath(__file__))
    proto_file = os.path.join(proto_dir, "consensus.proto")
    
    # Check if proto file exists
    if not os.path.exists(proto_file):
        print(f"Error: Proto file not found at {proto_file}")
        return False
    
    # Command to generate Python files
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={proto_dir}",
        f"--grpc_python_out={proto_dir}",
        proto_file
    ]
    
    try:
        print("Generating gRPC Python files...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("Successfully generated:")
        print(f"  - {proto_dir}/consensus_pb2.py")
        print(f"  - {proto_dir}/consensus_pb2_grpc.py")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error generating proto files: {e}")
        print(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = generate_proto_files()
    sys.exit(0 if success else 1)