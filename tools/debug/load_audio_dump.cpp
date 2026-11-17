// Debug tool: decode a WAV like librosa.load(sr=22050, mono=True) and write raw float32 samples.
//   load_audio_dump <in.wav> <out.f32>
//   load_audio_dump <in.f32> <out.f32> <in_rate>     resample raw mono float32 only
#include "bpp/bpp.h"
#include "resample.h"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    std::vector<float> audio;
    if (argc > 3) {
        std::ifstream f(argv[1], std::ios::binary);
        std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        std::vector<float> in(b.size() / 4);
        std::copy(b.begin(), b.begin() + in.size() * 4, reinterpret_cast<char*>(in.data()));
        audio = bpp::detail::resample_to_model_rate(in, static_cast<uint32_t>(std::stoul(argv[3])));
    } else {
        audio = bpp::load_audio(argv[1]);
    }
    std::ofstream(argv[2], std::ios::binary).write(reinterpret_cast<const char*>(audio.data()), audio.size() * 4);
    std::printf("%zu samples\n", audio.size());
    return 0;
}
