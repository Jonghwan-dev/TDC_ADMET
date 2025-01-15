import torch
import dgl
import dgllife

def check_cuda_availability():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("Number of GPUs:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print("Current device ID:", torch.cuda.current_device())
        print("DGL version:", dgl.__version__)
        print("DGLLife version:", dgllife.__version__)
    else:
        print("No CUDA-compatible GPU detected.")
        print("DGL version:", dgl.__version__)
        print("DGLLife version:", dgllife.__version__)

if __name__ == "__main__":
    check_cuda_availability()
