import numpy as np

# Load the .npy file
kernel_data = np.load("model/kernel.npy")

# Start C++ header content
header_content = '''#pragma once

#include <complex>
#include <array>

namespace amt {
namespace weights {

constexpr std::size_t KERNEL_N_BINS = %d;
constexpr std::size_t KERNEL_LENGTH = %d;

constexpr std::array<std::complex<float>, KERNEL_N_BINS * KERNEL_LENGTH> kernel_weights_array = {{
''' % (kernel_data.shape[0], kernel_data.shape[1])

# Add data
for i in range(kernel_data.shape[0]):
    for j in range(kernel_data.shape[1]):
        complex_val = kernel_data[i, j]
        header_content += "    std::complex<float>(%.9ef, %.9ef),\n" % (np.real(complex_val), np.imag(complex_val))

# End C++ header content
header_content += '''}};

} // namespace weights
} // namespace amt
'''

# Write to header file
with open("src/kernel_weights.h", "w") as f:
    f.write(header_content)

print("kernel_weights.h generated successfully.")
