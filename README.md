# FMCW-RADAR-PROJECT

This is my project repository for building an FMCW radar system. I am using 4 reference repositories to study and use as a base for the work. I've added them here as submodules so everything stays in one place and is easy to track.

The repos cover different aspects of FMCW radar — simulation in MATLAB, hardware firmware for TI mmWave chips, Python-based simulation, and target detection. I'll be going through each of these as the project moves forward and building on top of them.

---

## Reference Repositories (provided by supervisor)

### 1. FMCW-MIMO-Radar-Simulation
[ekurtgl/FMCW-MIMO-Radar-Simulation](https://github.com/ekurtgl/FMCW-MIMO-Radar-Simulation)

MATLAB simulation of a FMCW MIMO radar. Covers range, velocity and angle estimation, CA-CFAR detection, and angle spectrum using FFT and MUSIC. Added for reference on the simulation and signal processing side of the project.

---

### 2. fmcw-RADAR
[0xastro/fmcw-RADAR](https://github.com/0xastro/fmcw-RADAR)

Hardware firmware for the TI AWR1843 mmWave chip (77–81 GHz). Implements the full DSP chain — Range FFT, Doppler FFT, CFAR, DBSCAN clustering, and Extended Kalman Filter tracking. Added for reference on the hardware implementation side.

---

### 3. radarsimpy
[radarsimx/radarsimpy](https://github.com/radarsimx/radarsimpy)

Python and C++ based radar simulator. Supports FMCW and other waveforms, 3D scene simulation, DoA estimation, beamforming, and CFAR. Added as a reference for Python-based simulation and testing during the project.

---

### 4. radar-target-generation-and-detection
[davidscmx/radar-target-generation-and-detection](https://github.com/davidscmx/radar-target-generation-and-detection)

MATLAB/Octave project covering FMCW waveform generation, beat signal computation, Range FFT, Range-Doppler map, and CA-CFAR detection. Added as a reference for understanding the basic signal processing pipeline step by step.

---

## Status

Project is in the early stages. Will update this repo as the project develops.
