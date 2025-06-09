#pragma once

#include "typedef.h"
#include "layer.h"
#include <vector>
#include <string>

void loadDefaultKernel(Matrixcf &kernel);

void loadDefaultLowPassFilter( Vectorf &filter_kernel);

void getLayers(std::vector<Layer*> &layers, std::string model_name);