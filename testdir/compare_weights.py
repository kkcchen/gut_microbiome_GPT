import torch

# Load your model definition (replace with your actual model)

# Load saved weights into model_saved


def compare_weights(path1, path2):
    weights = torch.load(path1)
    weights1 = torch.load(path2)
    # Compare parameters
    for key in weights:
        if not torch.equal(weights[key], weights1[key]):
            abs_diff = (weights[key] - weights1[key]).abs()
            mean_diff = abs_diff.mean().item()
            max_diff = abs_diff.max().item()

            print(f"❌ Mismatch found in parameter: {key}")
            print(f"   Mean abs diff: {mean_diff:.6f}")
            print(f"   Max abs diff:  {max_diff:.6f}")
            max_val = max(weights[key].max().item(), weights1[key].max().item())
            min_val = min(weights[key].min().item(), weights1[key].min().item())
            print(f"{key} max value: {max_val}, min value: {min_val}")


def compare_data_states(path1, path2):
    data_state_1 = torch.load(path1)
    data_state_2 = torch.load(path2)

    keys = ["train_data_dict", "valid_data_dict"]
    subkeys = ["taxa_ids", "values"]

    all_match = True
    for key in keys:
        print(f"\nChecking {key}...")
        for subkey in subkeys:
            v1 = data_state_1[key][subkey]
            v2 = data_state_2[key][subkey]

            if isinstance(v1, torch.Tensor):
                equal = torch.equal(v1, v2)
            else:
                equal = v1 == v2

            if not equal:
                print(f"  MISMATCH in {key} -> {subkey}")
                all_match = False
            else:
                print(f"  Match for {key} -> {subkey}")

    if all_match:
        print("\n✅ Data splits are identical.")
    else:
        print("\n❌ Data splits differ.")
        
if __name__ == "__main__":
    # path1 = None
    # path2 = None
    # compare_weights(path1, path2)
    
    path3 = "experiment_saves/finetune/test_valid_long2/data_checkpoint/data_state.pt"
    path4 = "experiment_saves/finetune/test_valid_long4/data_checkpoint/data_state.pt"
    compare_data_states(path3, path4)