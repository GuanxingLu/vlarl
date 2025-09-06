"""
Simple test to verify GPU configuration and Ray setup.
This can be run independently to check if the environment is set up correctly.
"""

import os
import ray
import torch

def test_gpu_setup():
    """Test GPU configuration"""
    print("=== GPU Configuration Test ===")
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")
    
    # Check MUJOCO_EGL_DEVICE_ID
    mujoco_device = os.environ.get("MUJOCO_EGL_DEVICE_ID", "Not set")
    print(f"MUJOCO_EGL_DEVICE_ID: {mujoco_device}")
    
    # Check PyTorch CUDA availability
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    print()

def test_ray_setup():
    """Test Ray configuration"""
    print("=== Ray Configuration Test ===")
    
    try:
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                num_gpus=4,
                num_cpus=8,
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=False
            )
        
        # Check Ray cluster resources
        resources = ray.cluster_resources()
        print(f"Ray cluster resources: {resources}")
        
        # Test a simple Ray remote function
        @ray.remote(num_gpus=1)
        def test_gpu_worker(worker_id):
            return {
                "worker_id": worker_id,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
                "available_gpus": ray.get_gpu_ids() if hasattr(ray, 'get_gpu_ids') else "Not available"
            }
        
        # Create 4 workers
        workers = [test_gpu_worker.remote(i) for i in range(4)]
        results = ray.get(workers)
        
        print("\nRay GPU worker results:")
        for result in results:
            print(f"  Worker {result['worker_id']}: {result}")
        
        ray.shutdown()
        print("Ray test completed successfully!")
        
    except Exception as e:
        print(f"Ray test failed: {e}")
        if ray.is_initialized():
            ray.shutdown()
    
    print()

def main():
    """Run all configuration tests"""
    print("VLA-RL Environment Configuration Test")
    print("=" * 50)
    
    # Set environment variables for testing
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,7"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "1,2,3,7"
    
    test_gpu_setup()
    test_ray_setup()
    
    print("Configuration test completed!")

if __name__ == "__main__":
    main()
