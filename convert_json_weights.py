import json
import os

def get_vector_float_init_string(vector_data, var_name_logging):
    # print(f"    Generating C++ array string for: {var_name_logging} with {len(vector_data)} elements")
    if not isinstance(vector_data, list) or not vector_data:
        print(f"    WARNING: Empty or invalid vector_data for {var_name_logging}. Using default empty initializer.")
        return "{{}}" # Valid empty C++ array
    return "{{\n  " + ",\n  ".join([f"{float(val):.9e}f" for val in vector_data]) + "\n}}"

def generate_cpp_array_for_conv_weights_SIMPLIFIED(weights_data, conv_idx_str, layer_json_idx_str, model_name_for_log):
    print(f"  Model {model_name_for_log}, L{layer_json_idx_str}, CONV Idx {conv_idx_str}: Attempting to generate SIMPLIFIED weights array.")
    if not isinstance(weights_data, list) or not weights_data:
        print(f"  Model {model_name_for_log}, L{layer_json_idx_str}, CONV Idx {conv_idx_str}: Weights data is not a list or is empty. Skipping.")
        return ""

    try:
        d0 = len(weights_data)
        if d0 == 0: print(f"  CONV L{layer_json_idx_str}, C{conv_idx_str}: d0 is 0"); return ""
        d1 = len(weights_data[0])
        if d1 == 0: print(f"  CONV L{layer_json_idx_str}, C{conv_idx_str}: d1 is 0"); return ""
        d2 = len(weights_data[0][0])
        if d2 == 0: print(f"  CONV L{layer_json_idx_str}, C{conv_idx_str}: d2 is 0"); return ""
        d3 = len(weights_data[0][0][0])
        if d3 == 0: print(f"  CONV L{layer_json_idx_str}, C{conv_idx_str}: d3 is 0"); return ""
    except Exception as e:
        print(f"  Error accessing dimensions for CONV L{layer_json_idx_str}, C{conv_idx_str}: {e}. Skipping.")
        return ""

    content = f"// Conv2D Layer {layer_json_idx_str} (conv_idx {conv_idx_str}) Weights (SIMPLIFIED)\n"
    # Correct C++ std::array initialization: std::array<type, size> name = {{elements}};
    # For nested, it's std::array<std::array<float, D3>, D2> etc.
    # The {{ }} are for aggregate initialization.
    # Python f-string {{{{ -> C++ {{
    content += f"constexpr std::array<std::array<std::array<std::array<float, {d3}>, {d2}>, {d1}>, {d0}> conv{conv_idx_str}_weights = {{{{ \n" # Outermost {{

    for i0 in range(d0): # Output Filters
        content += "  {{ // Start d0 element\n"
        for i1 in range(d1): # Input Filters
            content += "    {{ // Start d1 element\n"
            for i2 in range(d2): # Kernel Height
                content += "      {{ // Start d2 element\n"
                # Innermost elements (Kernel Width)
                elements = []
                for i3 in range(d3):
                    try:
                        elements.append(f"{float(weights_data[i0][i1][i2][i3]):.9e}f")
                    except (IndexError, TypeError) as e:
                        print(f"Error accessing element [{i0}][{i1}][{i2}][{i3}]: {e}")
                        elements.append("0.0f /*ERROR*/") # Placeholder for error
                content += ", ".join(elements)
                content += "}}" # End d2 element
                if i2 < d2 - 1: content += ","
                content += "\n"
            content += "    }}" # End d1 element
            if i1 < d1 - 1: content += ","
            content += "\n"
        content += "  }}" # End d0 element
        if i0 < d0 - 1: content += ","
        content += "\n"
    # Python f-string }}}} -> C++ }};
    content += "}}}}; // Outermost }}\n\n"
    print(f"  Model {model_name_for_log}, L{layer_json_idx_str}, CONV Idx {conv_idx_str}: Successfully generated SIMPLIFIED weights string.")
    return content

def generate_header_for_model(model_pascal_name_unique, model_json_path, model_orig_name_for_log, process_only_first_conv=False):
    # (Same as before, but calls generate_cpp_array_for_conv_weights_SIMPLIFIED)
    print(f"--- Generating header for model: {model_orig_name_for_log} ({model_pascal_name_unique}) ---")
    with open(model_json_path, 'r') as f:
        model_data = json.load(f)

    header_filename = f"src/{model_pascal_name_unique.lower()}_model_weights.h"
    header_content = f'''#pragma once
#include <array>

namespace amt {{
namespace weights {{
namespace {model_pascal_name_unique} {{

'''
    print(f"  Outputting to: {header_filename}")
    print(f"  Using namespace: amt::weights::{model_pascal_name_unique}")

    conv_counter = 0
    # bn_counter = 0 # BatchNorm processing removed for this focused subtask

    for layer_idx, layer_data in enumerate(model_data["layers"]):
        layer_type = layer_data["type"]
        # print(f"  Model {model_orig_name_for_log}, Processing Layer {layer_idx}, Type: {layer_type}")

        if layer_type == "conv2d":
            if process_only_first_conv and conv_counter > 0:
                print(f"    L{layer_idx} CONV: Skipping due to process_only_first_conv flag.")
                # conv_counter +=1 # No, counter should only increment if processed or skipped with intent to not process
                continue # Skip this layer if flag is set and it's not the first conv layer

            weights_list = layer_data.get("weights", [])
            if not weights_list or len(weights_list) < 2:
                print(f"    L{layer_idx} CONV C{conv_counter}: 'weights' array missing or too short. Skipping.")
                conv_counter +=1
                continue

            weights_data = weights_list[0]
            biases_data = weights_list[1]

            # Use the new simplified/corrected function
            conv_weights_str = generate_cpp_array_for_conv_weights_SIMPLIFIED(weights_data, str(conv_counter), str(layer_idx), model_orig_name_for_log)
            if conv_weights_str: # Check if string is not empty (i.e., not skipped due to error)
                header_content += conv_weights_str
                print(f"    L{layer_idx} CONV C{conv_counter}: Generating biases.")
                header_content += f"// Conv2D Layer {layer_idx} (conv_idx {conv_counter}) Biases\n"
                header_content += f"constexpr std::array<float, {len(biases_data)}> conv{conv_counter}_biases = "
                # Python f-string {{{{ -> C++ {{ for outer brace of std::array
                header_content += get_vector_float_init_string(biases_data, f"conv{conv_counter}_biases") + ";\n\n"
            conv_counter += 1

        # BatchNorm processing is removed for this subtask to focus on Conv2D syntax
        # elif layer_type == "batchnorm2d":
        #    ...

    header_content += f"}} // namespace {model_pascal_name_unique}\n"
    header_content += f"}} // namespace weights\n"
    header_content += f"}} // namespace amt\n"

    with open(header_filename, "w") as f:
        f.write(header_content)
    print(f"  SUCCESS: {header_filename} generated for model {model_orig_name_for_log}.")
    print(f"--- Finished model: {model_orig_name_for_log} ---\n")

# --- Main script execution ---
# ONLY PROCESS CNNContour and ONLY its first conv layer for this subtask
model_info = {"unique_name": "CNNContour", "json_path": "model/cnn_contour_model.json", "orig_name": "Contour"}
generate_header_for_model(model_info["unique_name"], model_info["json_path"], model_info["orig_name"], process_only_first_conv=True)

# Create dummy files for other models to prevent include errors in loader.cpp for this specific test
other_models = [
    {"unique_name": "CNNNote", "json_path": "model/cnn_note_model.json", "orig_name": "Note"},
    {"unique_name": "CNNOnset1", "json_path": "model/cnn_onset_1_model.json", "orig_name": "Onset Input"},
    {"unique_name": "CNNOnset2", "json_path": "model/cnn_onset_2_model.json", "orig_name": "Onset Output"}
]
for m_info in other_models:
    dummy_header_filename = f"src/{m_info['unique_name'].lower()}_model_weights.h"
    with open(dummy_header_filename, "w") as f:
        f.write(f'''#pragma once
#include <array>
namespace amt {{ namespace weights {{ namespace {m_info['unique_name']} {{
// Dummy file for compilation test
}} }} }}
        ''')
    print(f"  Generated dummy header: {dummy_header_filename}")

print("Targeted JSON model weights conversion attempt finished.")
