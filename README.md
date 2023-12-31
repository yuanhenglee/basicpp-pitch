# BasiCPP Pitch: A C++ implementation for AMT(Automatic Music Transcription)

BasiCPP Pitch is a instrument-agnostic and polyphonic capable AMT(Automatic
Music Transcription) library written in C++.

## Basic Information

BasiCPP Pitch is an instrument-agnostic and polyphonic-capable AMT (Automatic
Music Transcription) library written in C++.

- github repository:[repo](https://github.com/yuanhenglee/basicpp-pitch)

Provide any compatible audio file, the library will generate a MIDI file with
the notes it detected. The library also provides Python API, which is
implemented by pybind11, to make it easier to use. 

The AMT model we used is from Spotify's
[basic-pitch](https://github.com/spotify/basic-pitch). More information about
the model can be found in the research paper, [A Lightweight Instrument-Agnostic
Model for Polyphonic Note Transcription and Multipitch
Estimation](https://arxiv.org/abs/2203.09893).

## How to build

```bash
./build.sh
```
The provided script will build the dynamic library `python/BasiCPP_Pitch.so` for Python API and the executable `bin/run` for C++ API.
To building the .so or the executable separately, you can use flags '-p' and '-e' respectively.

```bash
./build.sh -h

# Usage: cmd [-p] [-e]
#   -p: build python module
#   -e: build executable
#   -t: run tests, only valid when python module is built
#   -g: enable gprof profiling
```

## Run the example

```bash
# C++ example
./bin/run
# Python example
python3 python/run.py
```

## Problem to Solve

In music information retrieval (MIR), Automatic Music Transcription (AMT) aims
to convert raw audio recordings into symbolic representations like sheet music
or MIDI files.

One of the significant challenges in AMT is accurately transcribing polyphonic
audio, where multiple notes are played simultaneously. A practical solution
involves audio preprocessing techniques, e.g., Constant-Q Transform (CQT), to
represent audio in the frequency domain. This specialized transform can capture
the harmonic structure of music, which is vital for polyphonic transcription. By
stacking harmonics, we create a comprehensive frequency representation for each
time frame, enabling the subsequent steps to better discern individual notes in
the presence of harmonically rich audio. We employ a Convolutional Neural
Network (CNN) architecture to generate notes from the preprocessed audio frames. 

## Prospective Users

Music producers and engineers can employ it to transcribe audio recordings into
MIDI files, facilitating tasks like remixing and editing. Music researchers and
analysts can conduct in-depth compositional studies, while educators and
students can use it for ear training and composition exercises.

## System Architecture

![image](https://github.com/yuanhenglee/basicpp-pitch/blob/master/pics/NSD_project_flowchart.drawio.png)

### Input

The system takes an audio file that needs to be transcribed. as input.
The file can be in a standard format, such as WAV or MP3. 

### Harmonic stacking using constant-Q transform. 

The audio data undergoes preprocessing, beginning with applying the Constant-Q
Transform (CQT). This transforms the audio from the time domain to the frequency
domain, capturing the harmonic content crucial for polyphonic transcription.
Harmonic stacking is then applied to create a comprehensive frequency
representation for each time frame. This is a critical step for distinguishing
individual notes in harmonically rich audio. 

### Inference pre-trained CNN for note generation.

The preprocessed data is fed into a Convolutional Neural Network
(CNN) architecture. This trained model analyzes the frequency representations of
the audio frames to generate note information. The CNN is capable of accurately
identifying pitch information for each frame, allowing it to transcribe
polyphonic audio.

### Post-processing & Alignment

Post-processing steps are performed to refine the detected notes. This involves
tasks like note duration estimation, handling overlapping frequencies, and
convert vector-like transcription into MIDI. Note alignment ensures that the
generated notes are correctly timed, accurately representing the musical
content.

### Output

The system produces a MIDI file as output containing the transcribed musical
notes. 

## API Description

BasiCPP Pitch provides a user-friendly API for both C++ and Python, allowing
developers to integrate the AMT capabilities into their applications.

### C++ Example

```cpp
#include "amtModel.h"
#include "loader.h"

int main(int argc, char** argv) {
    
    auto audio = getExampleAudio();

    auto model = amtModel();

    auto notes = model.transcribeAudio(audio);

    return 0;
}
```
See `src/main.cpp` for more details.

### Python Example

```python
import BasiCPP_Pitch

# Load the example audio
audio = getExampleAudio()

# Initialize the model
model = BasiCPP_Pitch.amtModel()

# Transcribe the audio and generate the MIDI file
notes = model.transcribeAudio(audio)
midi = note2midi(notes)
midi.write(midi_file_path)
```
See `python/run.py` for more details.

## Engineering Infrastructure

1. Automatic build system: `CMake`
2. Version control: `Git`
3. Testing framework: `Pytest`
4. Documentation: `README`, `PyDoc`

## Schedule

* Week 1 (10/30):
    - [ ] Set up the environment and project structure
    - [ ] Test the pure Python prototype
    - [ ] Learning in-depth about CQT
* Week 2 (11/06):
    - [ ] Implement CQT ( in C++ and wrap it with pybind11 )
    - [ ] Write tests for CQT correctness
* Week 3 (11/13):
    - [ ] Implement the initialization of the CNN architecture
    - [ ] Implement the inference function ( in C++ and wrap it with pybind11 )
* Week 4 (11/20):
    - [ ] Gather the pre-trained CNN model weights in desired format
    - [ ] Implement the weight loader for the CNN model in C++
    - [ ] Write tests for the note generation
* Week 5 (11/27):
    - [ ] Implement the post-processing steps
    - [ ] Write tests for the post-processing steps
    - [ ] Implement the note alignment
    - [ ] Write tests for the note alignment
* Week 6 (12/04):
    - [ ] Try to improve the inference speed
        - [ ] XSIMD
        - [ ] Eigen
        - [ ] OpenMP
        - [ ] ...
* Week 7 (12/11):
    - [ ] Write documentation
    - [ ] Write examples
* Week 8 (12/18):
    - [ ] Refine the project structure
    - [ ] Prepare for the presentation


## References

- [basic-pitch](https://github.com/spotify/basic-pitch/tree/main)
- [A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation](https://arxiv.org/abs/2203.09893)
