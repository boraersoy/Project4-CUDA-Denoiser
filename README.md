CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Bora Ersoy
* Tested on: 13th Gen Intel(R) Core(TM) i7-13700HX, 2100 Mhz, 16 Core(s), 24 Logical Processor(s), RTX 4060 8GB AD107

### CUDA DENOISER
![](img/coverphoto.png)

This project is a implementation of the [Edge-Avoiding À-Trous Wavelet Transform for fast Global Illumination Filtering](https://jo.dreggn.org/home/2010_atrous.pdf) into a CUDA pathtracer.
This paper introduces an edge-aware À-Trous wavelet filter for fast denoising of noisy Monte Carlo global illumination images. By combining a multi-scale à-trous convolution with edge-stopping weights based on geometry and illumination differences, the method preserves sharp features like shadows and edges while smoothing noise and runs fast enough for interactive rendering. It lets a renderer produce smooth indirect lighting with far fewer samples than standard path tracing.

### Features
  - Geometry Buffer visualisation
  - Comparison with simple blur and edge avoiding blur
  - Filter size and weight parameters
  - Integration with main path tracer for more complex scenes
  - Performance impacts of the method

### Geometry Buffer

Geometry buffer is visualized here. You can see positions and normals  of the scene by clicking show PBuffer and NBuffer checkboxes. 

| Positions | Normals |
| ---- | ----|
|![](img/normalbuffer.png)|![](img/positionbuffer.png)|

### Comparison with simple blur and edge avoiding blur
Cornell Ceiling Light scene with 10 samples, color weight = 0.522, normal weight = 0.177, position weight = 0.720 with filter size 88
| Noisy Image | Simple Blur | Edge avoding blur |
| ---- | ----| ---- |
|![](img/10samplesnoblur.png)|![](img/10samplessimpleblur.png)|![](img/10sampleedgeavoidingblur.png)|

