import numpy as np

# Load the .npy file
lowpass_data = np.load("model/lowpass_filter.npy")

# Start C++ header content
header_content = '''#pragma once

#include <array>

namespace amt {
namespace weights {

constexpr std::size_t LOWPASS_FILTER_LENGTH = %d;

constexpr std::array<float, LOWPASS_FILTER_LENGTH> lowpass_filter_weights_array = {{
''' % (lowpass_data.shape[0])

# Add data
for i in range(lowpass_data.shape[0]):
    header_content += "    %.9ef,\n" % lowpass_data[i]

# End C++ header content
header_content += '''}};

} // namespace weights
} // namespace amt
'''

# Write to header file
with open("src/lowpass_filter_weights.h", "w") as f:
    f.write(header_content)

print("lowpass_filter_weights.h generated successfully.")
