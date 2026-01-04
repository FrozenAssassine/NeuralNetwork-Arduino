import torch
import numpy as np

# Load trained weights
state_dict = torch.load("xor_model_weights.pth", map_location="cpu")

HEADER_FILE = "nn_trained.h"

# Flatten arrays


def flatten_tensor(tensor):
    return tensor.detach().cpu().numpy().flatten()


def array_to_c(name, array):
    arr_str = ", ".join([f"{v:.6f}f" for v in array])
    return f"float {name}[{len(array)}] = {{ {arr_str} }};\n"


with open(HEADER_FILE, "w") as f:
    f.write("#pragma once\n\n")
    f.write("#include \"nn/layerData.h\"\n\n")

    meta_entries = []

    # add the input layer only with a given size
    input_size = state_dict['fc1.weight'].shape[1]
    meta_entries.append(
        f"{{nullptr, nullptr, 0, {input_size} }}")

    layer_idx = 0
    while f"fc{layer_idx+1}.weight" in state_dict:
        w_name = f"layer{layer_idx}_weights"
        b_name = f"layer{layer_idx}_bias"

        weights = flatten_tensor(state_dict[f"fc{layer_idx+1}.weight"])
        bias = flatten_tensor(state_dict[f"fc{layer_idx+1}.bias"])

        f.write(array_to_c(w_name, weights))
        f.write(array_to_c(b_name, bias))
        f.write("\n")

        # Meta entry
        input_size = state_dict[f"fc{layer_idx+1}.weight"].shape[1]
        output_size = state_dict[f"fc{layer_idx+1}.weight"].shape[0]
        meta_entries.append(
            f"{{ {w_name}, {b_name}, {input_size}, {output_size} }}")

        layer_idx += 1

    f.write("const LayerData nn_layers[] = {\n")
    f.write(",\n".join(meta_entries))
    f.write("\n};\n\n")
    f.write(f"const uint8_t nn_total_layers = {layer_idx + 1};\n")

print(f"Header generated: {HEADER_FILE}")
