#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo);
void showNBuffer(uchar4* pbo);
void showPBuffer(uchar4* pbo);
void denoise(uchar4* image, int filterSize, int iter, float c_phi,
    float n_phi, float p_phi);

void showImage(uchar4 *pbo, int iter);
