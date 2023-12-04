#pragma once

#include "typedef.h"
#include "layer.h"
#include <vector>


class CNN {
    public:

        CNN( int input_size, int output_size );

        ~CNN();
    
        // inference API for Eigen IO
        void forward( const float* input, float* output );

        std::string get_name() const;
    
        int _input_size;
        int _output_size;
        std::vector<Layer*> _layers;
};

class ContourCNN : public CNN {
    public:

        ContourCNN(); 

};