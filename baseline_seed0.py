import torch
import numpy as np
import random
import airbench94_muon as ab

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Note: Setting these ensures better reproducibility at a slight cost to performance.

def main():
    seed = 0
    set_seed(seed)
    
    print(f"Running baseline with seed {seed}, no warmup, and default compilation mode.")
    
    # Initialize model
    model = ab.CifarNet().cuda().to(memory_format=torch.channels_last)
    
    # Compile with mode="default" as requested
    model = torch.compile(model, mode="default")
    
    # Print header
    ab.print_columns(ab.logging_columns_list, is_head=True)
    
    # Skip warmup and run a single trial with seed 0
    # main() in airbench94_muon returns the tta_val_acc
    final_acc = ab.main(0, model)
    
    print("\n" + "="*30)
    print(f"Final TTA Accuracy: {final_acc:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()
