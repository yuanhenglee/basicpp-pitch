// #pragma once // loader.cpp should not have pragma once

#include "loader.h"
// Remove json.hpp and cnpy.h as they are no longer used for direct weight loading here
// #include "json.hpp"
// #include "cnpy.h"
#include <fstream> // Kept for now, might be removed if no file ops remain

// Include the new weight headers
#include "kernel_weights.h"
#include "lowpass_filter_weights.h"
#include "cnncontour_model_weights.h" // Corrected filenames from previous step
#include "cnnnote_model_weights.h"
#include "cnnonset1_model_weights.h"
#include "cnnonset2_model_weights.h"

// Layer class includes (assuming Conv2D, ReLU, Sigmoid are defined in layer.h or similar)
#include "layer.h"


// using json = nlohmann::json; // No longer using json here for weights

void loadDefaultKernel(Matrixcf &kernel) {
    // kernel is n_bins x kernel_length
    // kernel_weights_array is a flat std::array
    kernel.resize(amt::weights::KERNEL_N_BINS, amt::weights::KERNEL_LENGTH);
    for (std::size_t i = 0; i < amt::weights::KERNEL_N_BINS; ++i) {
        for (std::size_t j = 0; j < amt::weights::KERNEL_LENGTH; ++j) {
            kernel(i, j) = amt::weights::kernel_weights_array[i * amt::weights::KERNEL_LENGTH + j];
        }
    }
}

void loadDefaultLowPassFilter( Vectorf &filter_kernel) {
    filter_kernel.resize(amt::weights::LOWPASS_FILTER_LENGTH);
    for (std::size_t i = 0; i < amt::weights::LOWPASS_FILTER_LENGTH; ++i) {
        filter_kernel(i) = amt::weights::lowpass_filter_weights_array[i];
    }
}

// getModelPath is removed as model paths are no longer needed for weight loading in this file.

void getLayers( std::vector<Layer*>& layers, std::string model_name ) {
    layers.clear();
    if (model_name == "Contour") {
        // Attempt to access only the first conv layer's weights and biases for CNNContour
        // This is to check if the generated cnncontour_model_weights.h is syntactically correct
        // and if the specific variables conv0_weights and conv0_biases are accessible.
        // Note: The actual use of these variables (e.g. printing size or an element)
        // might be optimized out if not careful, but referencing them should be enough for the compiler.
        [[maybe_unused]] const auto& test_weights = amt::weights::CNNContour::conv0_weights;
        [[maybe_unused]] const auto& test_biases = amt::weights::CNNContour::conv0_biases;
        // std::cout << "Test: Contour conv0_weights accessed. First element of bias: " << amt::weights::CNNContour::conv0_biases[0] << std::endl;
    }
    // Other models are not processed in this specific test's getLayers
}

// Old loadWeights function is removed.
// Old getExampleAudio function is removed.