// Standard MIDI File writer reproducing basic-pitch's note_events_to_midi() as serialised by
// pretty_midi 0.2.x + mido 1.3 (sorting rules, tick quantisation and running status included).
#include "bpp/bpp.h"

#include "constants.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>

namespace bpp {

namespace {

constexpr int kResolution = 220;      // pretty_midi default ticks per beat
constexpr int kElectricPiano1 = 4;    // instrument_name_to_program("Electric Piano 1")
constexpr int kPitchBendTicks = 8192;

enum EventType { kSetTempo, kTimeSignature, kProgramChange, kPitchWheel, kNoteOn, kEndOfTrack };

struct Event {
    long long tick;
    EventType type;
    int a = 0, b = 0;  // note/velocity, program, or pitch

    // pretty_midi's event_compare secondary key
    long long secondary() const {
        switch (type) {
            case kSetTempo: return 1 * 256 * 256;
            case kTimeSignature: return 2 * 256 * 256;
            case kProgramChange: return 6 * 256 * 256;
            case kPitchWheel: return 7 * 256 * 256 + a;
            case kNoteOn: return 10 * 256 * 256 + a * 256 + b;
            case kEndOfTrack: return 11 * 256 * 256;
        }
        return 0;
    }
};

void sort_events(std::vector<Event>& ev) {
    std::stable_sort(ev.begin(), ev.end(), [](const Event& x, const Event& y) {
        if (x.tick != y.tick) return x.tick < y.tick;
        return x.secondary() < y.secondary();
    });
}

void put_varint(std::vector<uint8_t>& out, unsigned long long v) {
    uint8_t buf[10];
    int n = 0;
    buf[n++] = v & 0x7f;
    while (v >>= 7) buf[n++] = 0x80 | (v & 0x7f);
    while (n) out.push_back(buf[--n]);
}

void put_be(std::vector<uint8_t>& out, unsigned v, int bytes) {
    for (int i = bytes - 1; i >= 0; --i) out.push_back((v >> (8 * i)) & 0xff);
}

void write_track(std::vector<uint8_t>& file, const std::vector<Event>& ev, int channel, int tempo) {
    std::vector<uint8_t> data;
    int running = -1;
    long long prev = 0;
    for (const Event& e : ev) {
        put_varint(data, static_cast<unsigned long long>(e.tick - prev));
        prev = e.tick;
        auto channel_msg = [&](int status, std::initializer_list<int> bytes) {
            if (status != running) data.push_back(static_cast<uint8_t>(status));
            for (int v : bytes) data.push_back(static_cast<uint8_t>(v));
            running = status;
        };
        switch (e.type) {
            case kSetTempo:
                data.insert(data.end(), {0xff, 0x51, 0x03});
                put_be(data, tempo, 3);
                running = -1;
                break;
            case kTimeSignature:
                data.insert(data.end(), {0xff, 0x58, 0x04, 4, 2, 24, 8});
                running = -1;
                break;
            case kEndOfTrack:
                data.insert(data.end(), {0xff, 0x2f, 0x00});
                running = -1;
                break;
            case kProgramChange: channel_msg(0xc0 | channel, {e.a}); break;
            case kNoteOn: channel_msg(0x90 | channel, {e.a, e.b}); break;
            case kPitchWheel: {
                const int v = e.a + 8192;
                channel_msg(0xe0 | channel, {v & 0x7f, v >> 7});
                break;
            }
        }
    }
    file.insert(file.end(), {'M', 'T', 'r', 'k'});
    put_be(file, static_cast<unsigned>(data.size()), 4);
    file.insert(file.end(), data.begin(), data.end());
}

struct MidiNote {
    double start, end;
    int pitch;
    float amplitude;
    std::optional<std::vector<int>> bends;
};

}  // namespace

std::vector<uint8_t> notes_to_midi(const std::vector<NoteEvent>& notes, const NoteOptions& opt) {
    // pretty_midi / mido reject the same inputs (ZeroDivisionError, ValueError)
    if (!(opt.midi_tempo > 0) || !std::isfinite(opt.midi_tempo))
        throw std::invalid_argument("midi_tempo must be positive");
    for (const NoteEvent& n : notes) {
        if (!(n.amplitude >= 0.0f && n.amplitude <= 1.0f)) throw std::invalid_argument("note amplitude outside [0, 1]");
        if (n.pitch < 0 || n.pitch > 127) throw std::invalid_argument("note pitch outside [0, 127]");
    }
    std::vector<MidiNote> events;
    events.reserve(notes.size());
    for (const NoteEvent& n : notes) {
        MidiNote m{n.start_s, n.end_s, n.pitch, n.amplitude, std::nullopt};
        if (!n.pitch_bends.empty()) m.bends = n.pitch_bends;
        events.push_back(std::move(m));
    }

    if (!opt.multiple_pitch_bends) {
        // drop_overlapping_pitch_bends(): sorted() on (start, end, pitch, amplitude, bends) tuples
        std::stable_sort(events.begin(), events.end(), [](const MidiNote& x, const MidiNote& y) {
            return std::tie(x.start, x.end, x.pitch, x.amplitude) < std::tie(y.start, y.end, y.pitch, y.amplitude);
        });
        for (size_t i = 0; i + 1 < events.size(); ++i)
            for (size_t j = i + 1; j < events.size(); ++j) {
                if (events[j].start >= events[i].end) break;
                events[i].bends.reset();
                events[j].bends.reset();
            }
    }

    const double tick_scale = 60.0 / (opt.midi_tempo * kResolution);
    const int tempo = static_cast<int>(6e7 / (60.0 / (tick_scale * kResolution)));
    // PrettyMIDI.time_to_tick() for a file built from scratch (tick-to-time table == [0])
    auto to_tick = [&](double t) -> long long {
        if (!(t > 0.0)) return 0;
        return static_cast<long long>(std::nearbyint(t / tick_scale));
    };

    // instruments in insertion order: one per pitch with multiple_pitch_bends, else a single one
    std::vector<int> inst_keys;
    std::map<int, std::vector<Event>> inst_events;
    for (const MidiNote& m : events) {
        const int key = opt.multiple_pitch_bends ? m.pitch : 0;
        auto it = inst_events.find(key);
        if (it == inst_events.end()) {
            inst_keys.push_back(key);
            it = inst_events.emplace(key, std::vector<Event>{}).first;
        }
        std::vector<Event>& ev = it->second;
        // int(np.round(127 * amplitude)): NumPy 1.x promotes the float32 scalar product to float64
        const int velocity = static_cast<int>(std::nearbyint(127.0 * double(m.amplitude)));
        ev.push_back({to_tick(m.start), kNoteOn, m.pitch, velocity});
        ev.push_back({to_tick(m.end), kNoteOn, m.pitch, 0});
        if (!m.bends) continue;
        const std::vector<int>& b = *m.bends;
        const int num = static_cast<int>(b.size());
        // np.linspace(start, end, num): start + i * step, last sample pinned to end
        const double step = num > 1 ? (m.end - m.start) / double(num - 1) : 0.0;
        for (int i = 0; i < num; ++i) {
            double t;
            if (num == 1) t = m.start;
            else if (i == num - 1) t = m.end;
            else if (step == 0.0) t = (double(i) / double(num - 1)) * (m.end - m.start) + m.start;
            else t = double(i) * step + m.start;
            long long ticks = static_cast<long long>(std::nearbyint(double(b[i] * 4096) / 3.0));
            ticks = std::clamp<long long>(ticks, -kPitchBendTicks, kPitchBendTicks - 1);
            // pitch bends are appended after all notes of the instrument; keep them in a second pass
            ev.push_back({to_tick(t), kPitchWheel, static_cast<int>(ticks), 0});
        }
    }

    std::vector<uint8_t> file;
    file.insert(file.end(), {'M', 'T', 'h', 'd', 0, 0, 0, 6, 0, 1});
    put_be(file, static_cast<unsigned>(1 + inst_keys.size()), 2);
    put_be(file, kResolution, 2);

    std::vector<Event> timing = {{0, kTimeSignature}, {0, kSetTempo}};
    sort_events(timing);
    timing.push_back({timing.back().tick + 1, kEndOfTrack});
    write_track(file, timing, 0, tempo);

    static constexpr int kChannels[15] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15};
    for (size_t n = 0; n < inst_keys.size(); ++n) {
        const std::vector<Event>& raw = inst_events[inst_keys[n]];
        // pretty_midi appends all notes first, then all pitch bends
        std::vector<Event> track;
        track.reserve(raw.size() + 2);
        track.push_back({0, kProgramChange, kElectricPiano1});
        for (const Event& e : raw)
            if (e.type == kNoteOn) track.push_back(e);
        for (const Event& e : raw)
            if (e.type == kPitchWheel) track.push_back(e);
        sort_events(track);
        const std::vector<Event> sorted = track;  // Python zips over slice copies of the sorted list
        for (size_t i = 0; i + 1 < sorted.size(); ++i) {
            const Event &e1 = sorted[i], &e2 = sorted[i + 1];
            if (e1.tick == e2.tick && e1.type == kNoteOn && e2.type == kNoteOn && e1.a == e2.a && e1.b != 0 && e2.b == 0) {
                track[i] = e2;
                track[i + 1] = e1;
            }
        }
        track.push_back({track.back().tick + 1, kEndOfTrack});
        write_track(file, track, kChannels[n % 15], tempo);
    }
    return file;
}

}  // namespace bpp
