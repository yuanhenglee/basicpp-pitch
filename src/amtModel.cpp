#include "amtModel.h"
#include "utils.h"
#include <iostream>

// #define USE_OMP
#define USE_PTHREADS

#ifdef USE_OMP
#include <omp.h>
#endif

#define USE_PTHREADS
#ifdef USE_PTHREADS
// #include <pthread.h>
#include <thread>
#endif

amtModel::amtModel(): 
    _cqt(),
    _onset_input_cnn("Onset Input"),
    _onset_output_cnn("Onset Output"),
    _note_cnn("Note"),
    _contour_cnn("Contour") {

#if defined USE_OMP || defined USE_PTHREADS
    Eigen::initParallel();
#endif

}

void amtModel::reset() {
    _audio_len = 0;
    _Yp_buffer.clear();
    _Yn_buffer.clear();
    _Yo_buffer.clear();
}

std::vector<Note> amtModel::transcribeAudio( const Vectorf& audio ) {

    // reset the model
    reset();

    _audio_len = audio.size();
    auto audio_windowed = getWindowedAudio(audio);
    

#ifdef USE_OMP
    // reserve buffer size
    _Yp_buffer.resize(audio_windowed.size());
    _Yn_buffer.resize(audio_windowed.size());
    _Yo_buffer.resize(audio_windowed.size());
#pragma omp parallel for
    for ( int i = 0 ; i < audio_windowed.size() ; i++ ) {
        // inferenceFrame(audio_windowed[i]);
        VecMatrixf cqt = _cqt.cqtHarmonic(audio_windowed[i], true);
        VecMatrixf contour_out = _contour_cnn.forward(cqt);
        _Yp_buffer[i] = contour_out[0]; // Yp
        VecMatrixf note_out = _note_cnn.forward(contour_out);
        _Yn_buffer[i] = note_out[0]; // Yn
        VecMatrixf concat_buf = {note_out[0]};
        VecMatrixf onset_out = _onset_input_cnn.forward(cqt);
        concat_buf.insert(concat_buf.end(), onset_out.begin(), onset_out.end());
        VecMatrixf concat_out = _onset_output_cnn.forward(concat_buf);
        _Yo_buffer[i] = concat_out[0]; // Yo
    }
#elif defined USE_PTHREADS
    // reserve buffer size
    _Yp_buffer.resize(audio_windowed.size());
    _Yn_buffer.resize(audio_windowed.size());
    _Yo_buffer.resize(audio_windowed.size());
    const int max_threads = std::getenv("OMP_NUM_THREADS") ? atoi(std::getenv("OMP_NUM_THREADS")) : 1;
    std::vector<std::thread> threads(max_threads);
    std::vector<PthreadArg*> args_to_delete;
    // PthreadArg objects are dynamically allocated for each thread.
    // These objects need to be manually deallocated after the threads are joined to prevent memory leaks.
    // args_to_delete vector stores the pointers to these objects for later deallocation.
    for ( int i = 0 ; i < audio_windowed.size() ; ) {
        int used_threads = 0;
        args_to_delete.reserve(max_threads);
        while ( used_threads < max_threads && i < audio_windowed.size() ) {
            PthreadArg* arg = new PthreadArg;
            arg->audio = &audio_windowed[i];
            arg->idx = i;
            args_to_delete.push_back(arg); // Store pointer for deallocation
            threads[used_threads] = std::thread(&amtModel::inferenceFramePthread, this, arg);
            i++;
            used_threads++;
        }
        for ( int j = 0 ; j < used_threads ; j++ ) {
            threads[j].join();
        }
        // Deallocate PthreadArg objects
        for ( int j = 0 ; j < used_threads ; j++ ) {
            delete args_to_delete[j]; // Deallocate the PthreadArg object
        }
        args_to_delete.clear();
    }
    threads.clear();
#else
    for ( int i = 0 ; i < audio_windowed.size() ; i++ ) {
        inferenceFrame(audio_windowed[i]);
    }
#endif

    // concat 3 buffers
    Matrixf Yp = concatMatrices(_Yp_buffer, _audio_len);
    Matrixf Yn = concatMatrices(_Yn_buffer, _audio_len);
    Matrixf Yo = concatMatrices(_Yo_buffer, _audio_len);

    // convert to midi note events
    return modelOutput2Notes(Yp, Yn, Yo, true);
}

// input shape : (N_AUDIO_SAMPLES, N_BIN_CONTORU )
void amtModel::inferenceFrame( const Vectorf& x ) {
    VecMatrixf output;

    // compute harmonic stacking, shape : (n_harmonics, n_frames, n_bins)
    VecMatrixf cqt = _cqt.cqtHarmonic(x, true);

    VecMatrixf contour_out = _contour_cnn.forward(cqt);
    _Yp_buffer.push_back(contour_out[0]); // Yp

    VecMatrixf note_out = _note_cnn.forward(contour_out);
    _Yn_buffer.push_back(note_out[0]); // Yn

    VecMatrixf concat_buf = {note_out[0]};
    VecMatrixf onset_out = _onset_input_cnn.forward(cqt);
    concat_buf.insert(concat_buf.end(), onset_out.begin(), onset_out.end());

    VecMatrixf concat_out = _onset_output_cnn.forward(concat_buf);
    _Yo_buffer.push_back(concat_out[0]); // Yo

}

void amtModel::inferenceFramePthread( PthreadArg* arg ) {
    size_t idx = arg->idx;
    // compute harmonic stacking, shape : (n_harmonics, n_frames, n_bins)
    VecMatrixf cqt = _cqt.cqtHarmonic(*(arg->audio), true);

    VecMatrixf contour_out = _contour_cnn.forward(cqt);
    _Yp_buffer[idx] = contour_out[0]; // Yp

    VecMatrixf note_out = _note_cnn.forward(contour_out);
    _Yn_buffer[idx] = note_out[0]; // Yn

    VecMatrixf concat_buf = {note_out[0]};
    VecMatrixf onset_out = _onset_input_cnn.forward(cqt);
    concat_buf.insert(concat_buf.end(), onset_out.begin(), onset_out.end());

    VecMatrixf concat_out = _onset_output_cnn.forward(concat_buf);
    _Yo_buffer[idx] = concat_out[0]; // Yo
}

VecMatrixf amtModel::getOutput() {
    // concat 3 buffers
    Matrixf Yp = concatMatrices(_Yp_buffer, _audio_len);
    Matrixf Yn = concatMatrices(_Yn_buffer, _audio_len);
    Matrixf Yo = concatMatrices(_Yo_buffer, _audio_len);
    return {Yp, Yn, Yo};
}