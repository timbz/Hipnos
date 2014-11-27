#HIgh Performance Non Linear Optics Simulation 
#####Modelling and simulation of optical beamlines

##Motivation

Up to today there are no tools that support the modeling and simulation of optical beam lines in a sophisticated visual manner. Existing tools are costly and difficult to understand for the end user, especially when the beam lines become very complex. Most optical beam systems can be abstracted to a linear beam line, resulting in a reduced complexity and better understandability. It is furthermore possible to group functionally related elements together to reduce the complexity of the system.

##Description

This diploma thesis presents a simulation software for high performance lasers. The graphical interface provides the user with the ability to string together optical components to build simple beam lines. The propagation of light through the system can be simulated with the gaussian beam model and the fourier optics model. Intensive mathematical operations are sourced out from the main application into plugins. As a consequence of this design it is possible to implement different versions of the plugin interface tailored towards specific system configurations. It is further investigated how the parallel processing power of modern GPUs can be used to improve computation time.

##Results

###Propagation simulation of a laser beam using the gaussian beam model

![Propagation simulation of a laser beam using the gaussian beam model](https://raw.githubusercontent.com/timonbaetz/Hipnos/master/docs/img/screen1.png)

###Propagation simulation of a laser beam with fourier optics

![Propagation simulation of a laser beam with fourier optics](https://raw.githubusercontent.com/timonbaetz/Hipnos/master/docs/img/screen2.png)

## Dependecies

* Qt 4
* VTK 5

##Build:
0. Install the Qt SDK
1. Open hipnos.pro with QtCreator
2. Set the corresponding environment  variables depending on the plugins you want to build:
	- HIPNOS_BUILD_ACML_PLUGIN  
	- HIPNOS_BUILD_APPML_PLUGIN 
	- HIPNOS_BUILD_CBLAS_PLUGIN 
	- HIPNOS_BUILD_CUDA_PLUGIN 
	- HIPNOS_BUILD_GSL_PLUGIN 
3. Install dependencies:
	- Main application: VTK (Windows: dependencies included in the project)
	- HipnosACMLMathPlugin: AMD ACML
	- HipnosAPPMLMathPlugin: AMD APPML
	- HipnosCBlasMathPlugin: any CBlas compatible library, FFTW
	- HipnosCudaMathPlugin & HipnosCudaSinglePrecisionMathPlugin: CUDA SDK
	- HipnosGSLMathPlugin: GSL (Windows: dependencies included in the project)
4. For each plugin set the INCLUDEPATH and LIBS variables in the plugins/[PluginName]/[PluginName].pro file
5. Build
