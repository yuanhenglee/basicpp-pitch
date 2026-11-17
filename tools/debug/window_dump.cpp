// Debug tool: runs one analysis window and dumps every intermediate tensor as .npy files.
// usage: window_dump <window.f32 (43844 raw floats)> <out_dir>
#include "window_model.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace bpp;
using namespace bpp::detail;

static void npy(const std::string& path, const float* p, std::vector<int> shape) {
    std::string dims;
    size_t n = 1;
    for (int d : shape) dims += std::to_string(d) + ", ", n *= d;
    std::string h = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + dims + "), }";
    while ((10 + h.size() + 1) % 64) h += ' ';
    h += '\n';
    std::ofstream f(path, std::ios::binary);
    f.write("\x93NUMPY\x01\x00", 8);
    const unsigned short len = static_cast<unsigned short>(h.size());
    f.write(reinterpret_cast<const char*>(&len), 2);
    f.write(h.data(), h.size());
    f.write(reinterpret_cast<const char*>(p), n * sizeof(float));
}

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    std::vector<float> win(kWindowSamples);
    std::ifstream(argv[1], std::ios::binary).read(reinterpret_cast<char*>(win.data()), win.size() * 4);
    Workspace ws;
    std::vector<float> contour(kUsedFrames * kContourBins), note(kUsedFrames * kNoteBins), onset(kUsedFrames * kNoteBins);
    WindowModel::get().run(win.data(), ws, contour.data(), note.data(), onset.data());
    const std::string d = argv[2];
    npy(d + "/cqt_re.npy", ws.cqt_re.data(), {172, 309});
    npy(d + "/cqt_im.npy", ws.cqt_im.data(), {172, 309});
    npy(d + "/bn.npy", ws.logp.data(), {172, 309});
    npy(d + "/hs.npy", ws.hs.data(), {172, 302, 8});
    npy(d + "/c1.npy", ws.c1.data(), {8, 172, 272});
    npy(d + "/yp.npy", ws.yp.data(), {172, 268});
    npy(d + "/n1.npy", ws.n1.data(), {32, 172, 96});
    npy(d + "/cat.npy", ws.cat.data(), {33, 172, 96});
    npy(d + "/contour.npy", contour.data(), {142, 264});
    npy(d + "/note.npy", note.data(), {142, 88});
    npy(d + "/onset.npy", onset.data(), {142, 88});
    for (int o = 0; o < kOctaves; ++o) {
        int n = kWindowSamples;
        for (int i = 0; i < o; ++i) n = (n - 2) / 2 + 1;
        npy(d + "/sig" + std::to_string(o) + ".npy", ws.sig[o].data(), {n + 256});
    }
    return 0;
}
