# FMCW Radar Range Enhancement Project — Phase-wise README Collection

---

# Phase 1 — Environment Setup & Study

## Overview
The first phase of the project focuses on building the foundation required for the complete FMCW radar simulation pipeline. Instead of directly jumping into implementation, this stage is dedicated to understanding how FMCW radar systems behave, how beat signals are generated, and how radar data eventually becomes usable range information.

This phase also includes setting up the development environment, organizing reference materials, and studying the internal workflow of existing radar simulation repositories.

The goal here is simple:

> Build enough understanding to confidently design and modify a radar processing pipeline instead of treating it as a black box.

---

## Main Objectives

- Configure the complete simulation environment using Python and MATLAB
- Install and test radar simulation libraries such as `radarsimpy`
- Study multiple FMCW radar reference repositories
- Understand waveform generation and chirp behaviour
- Learn how beat signals are produced from reflected targets
- Understand FFT-based range extraction
- Build familiarity with Range-Doppler processing
- Create a reusable folder structure for future phases

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python | Core simulation and algorithm development |
| MATLAB | Signal processing analysis and visualization |
| radarsimpy | Radar scene and signal simulation |
| NumPy | Numerical operations |
| Matplotlib | Plotting and analysis |
| Jupyter Notebook | Rapid experimentation |

---

## What Was Studied

### FMCW Radar Fundamentals
The initial study focused on understanding how Frequency Modulated Continuous Wave radar differs from conventional pulsed radar systems.

Important concepts explored:

- Chirp slope
- Sweep bandwidth
- Beat frequency
- Round trip delay
- Range estimation
- Velocity estimation
- Range resolution
- Sampling constraints

Special attention was given to understanding the relationship between transmitted and received chirps.

---

### Beat Signal Understanding
A large portion of this phase involved learning how the received signal mixes with the transmitted waveform to generate the beat signal.

Key observations:

- Target distance affects beat frequency
- Moving targets introduce Doppler shift
- Multiple targets create multiple frequency components
- FFT converts beat frequencies into range peaks

This understanding becomes critical for every later stage of the project.

---

### FFT Processing Pipeline
The FFT pipeline was studied step-by-step rather than using prebuilt implementations.

Processing flow:

1. Chirp generation
2. Signal reflection from target
3. Mixing transmitted and received signals
4. Beat signal extraction
5. Windowing
6. FFT computation
7. Peak detection

This stage helped in understanding where noise, leakage, and false detections originate.

---

## Repository Structure Created

```text
project-root/
│
├── data/
├── notebooks/
├── simulations/
├── algorithms/
├── plots/
├── reports/
└── references/
```

The structure was intentionally designed to keep simulations, algorithms, plots, and documentation separate for easier scaling.

---

## Key Outcomes

By the end of this phase:

- The complete simulation environment was functional
- Radar signal flow became conceptually clear
- Reference repositories were analyzed and documented
- Basic waveform experiments were successfully executed
- The FFT-based range extraction pipeline was understood in detail

This phase laid the groundwork for building a custom radar simulation instead of relying entirely on external implementations.

---

# Phase 2 — Monostatic FMCW Baseline

## Overview
This phase focuses on building a complete monostatic FMCW radar simulation pipeline from scratch.

A monostatic radar uses the same location for transmission and reception. Since this configuration is comparatively simpler, it acts as the baseline model for validating signal generation, range processing, Doppler estimation, and detection algorithms.

The primary objective of this stage is to create a working reference model before introducing advanced bistatic geometry and enhancement algorithms.

---

## Main Objectives

- Generate FMCW chirp waveforms
- Simulate transmitted and reflected signals
- Produce beat signals for targets
- Perform Range FFT
- Generate Range-Doppler maps
- Implement CA-CFAR target detection
- Validate target estimation accuracy

---

## System Model

The monostatic radar setup includes:

- Single transmitter
- Single receiver
- Static or moving targets
- Controlled simulation environment
- Additive noise model

The transmitter and receiver remain colocated throughout the simulation.

---

## Signal Processing Pipeline

### 1. Waveform Generation
FMCW chirps were generated using configurable parameters:

- Bandwidth
- Sweep time
- Carrier frequency
- Sampling rate
- Chirp slope

Different parameter combinations were tested to study their effect on range resolution.

---

### 2. Beat Signal Generation
Reflected signals were delayed based on target distance.

The transmitted and received signals were mixed to generate the intermediate frequency beat signal.

This stage verified:

- Accurate delay modelling
- Frequency shifting behaviour
- Target-dependent beat frequencies

---

### 3. Range FFT
The beat signal was transformed into the frequency domain using FFT.

The FFT output produced clear peaks corresponding to target ranges.

Additional improvements explored:

- Windowing functions
- Noise suppression
- Peak sharpening
- FFT size variation

---

### 4. Range-Doppler Mapping
Multiple chirps were processed together to estimate both:

- Target range
- Target velocity

This stage introduced Doppler processing and helped visualize moving targets in two dimensions.

---

### 5. CA-CFAR Detection
Cell Averaging CFAR was implemented to perform adaptive target detection.

The detector dynamically adjusted thresholds based on surrounding noise conditions.

Important observations:

- Fixed thresholds fail in noisy environments
- CFAR improves detection reliability
- Guard cells reduce target leakage effects

---

## Outputs Generated

- FMCW chirp plots
- Beat signal visualizations
- Range FFT plots
- Range-Doppler heatmaps
- CFAR detection maps

---

## Key Outcomes

At the completion of this phase:

- A fully functional monostatic radar pipeline was created
- Range estimation became stable and measurable
- Doppler processing was validated
- CFAR-based detection was operational
- The baseline model became ready for future enhancement work

This phase effectively became the reference system against which all later improvements would be measured.

---

# Phase 3A — Bistatic Geometry

## Overview
This phase introduces bistatic radar geometry into the project.

Unlike monostatic radar systems, bistatic radar separates the transmitter and receiver into different physical locations. This changes the signal propagation path and introduces new geometric behaviour that can significantly influence detection performance.

The focus of this phase is to redesign the signal model so that it correctly handles bistatic propagation and geometry-aware range computation.

---

## Main Objectives

- Separate transmitter and receiver locations
- Implement bistatic range equations
- Update propagation modelling
- Study bistatic angle effects
- Simulate multiple baseline configurations
- Modify beat signal generation for bistatic paths

---

## Why Bistatic Radar?

Bistatic radar systems offer several practical and theoretical advantages:

- Better target visibility from alternate angles
- Reduced vulnerability to direct reflection loss
- Improved stealth target observation in some scenarios
- Enhanced spatial coverage
- Additional geometric diversity

This phase explores how geometry itself can contribute to range enhancement.

---

## Bistatic Geometry Model

The signal path now includes:

- Transmitter-to-target distance (Rt)
- Target-to-receiver distance (Rr)

The total propagation distance becomes:

```math
R_total = Rt + Rr
```

This fundamentally changes how delays and beat frequencies are calculated.

---

## Geometry-Aware Beat Signal

The beat signal generation logic was modified to account for:

- Independent transmitter location
- Independent receiver location
- Bistatic delay
- Angle-based reflections
- Variable baseline distances

This created a more physically realistic radar environment.

---

## Baseline Configuration Experiments

Different transmitter-receiver separations were tested to study:

- Detection sensitivity
- Signal strength variation
- Geometric coverage
- Bistatic angle influence
- Range estimation behaviour

The experiments helped identify configurations that improved target observability.

---

## Enhanced Radar Cross Section Analysis

Certain bistatic angles produced stronger reflected energy compared to the monostatic setup.

This observation highlighted an important insight:

> Geometry alone can improve target detectability even before introducing advanced signal processing algorithms.

---

## Outputs Generated

- Bistatic geometry diagrams
- Delay comparison plots
- Baseline configuration visualizations
- Bistatic beat signal plots
- Range response comparisons

---

## Key Outcomes

By the end of this phase:

- The monostatic pipeline was successfully converted into a bistatic system
- Geometry-aware propagation modelling became operational
- Multiple baseline experiments were completed
- Bistatic effects on target response were clearly observed
- The system became ready for algorithmic enhancement stages

---

# Phase 3B — Algorithms for Range Enhancement

## Overview
After establishing the bistatic geometry framework, this phase focuses on improving radar performance through advanced signal processing algorithms.

The goal is not just to detect targets, but to improve:

- Range resolution
- Detection probability
- Noise robustness
- Angular estimation accuracy
- Weak target visibility

This phase acts as the intelligence layer of the project.

---

## Main Objectives

- Improve range resolution
- Reduce false detections
- Enhance weak target visibility
- Increase signal-to-noise ratio
- Implement super-resolution techniques
- Explore sparse signal reconstruction methods

---

## Zero-Padding FFT

Zero-padding was introduced before FFT computation.

Benefits observed:

- Smoother spectral representation
- Improved peak interpolation
- Better visualization of target separation
- Cleaner range spectrum

Although zero-padding does not physically increase resolution, it improves interpretability and peak localization.

---

## MUSIC Super-Resolution Algorithm

The MUSIC algorithm was explored for high-resolution direction-of-arrival estimation.

Key features:

- Subspace-based processing
- Ability to separate closely spaced targets
- Higher angular resolution than conventional FFT methods

This stage introduced more advanced mathematical signal decomposition techniques.

---

## Improved CFAR Variants

Beyond standard CA-CFAR, additional detection methods were explored:

- OS-CFAR
- Adaptive thresholding
- Noise-aware detection strategies

The goal was to improve performance in cluttered or non-uniform noise environments.

---

## Coherent Signal Integration

Multiple radar returns were integrated coherently to improve overall signal strength.

Advantages:

- Better weak target visibility
- Higher SNR
- More stable detections
- Reduced random noise impact

This became particularly useful in low-reflection target scenarios.

---

## Sparse and Compressed Sensing Methods

Sparse reconstruction approaches were explored for extracting meaningful target information from limited observations.

This section focused on:

- Reduced sampling approaches
- Sparse recovery principles
- Efficient signal representation
- High-resolution reconstruction

These methods are especially relevant in modern radar research.

---

## Outputs Generated

- Enhanced FFT comparisons
- Super-resolution plots
- CFAR benchmarking results
- Integrated signal visualizations
- Sparse reconstruction outputs

---

## Key Outcomes

By the completion of this phase:

- Multiple range enhancement techniques were implemented
- Detection stability improved significantly
- Resolution performance increased
- Weak target handling became more reliable
- The project gained a strong algorithmic processing layer

---

# Phase 4 — Combining Bistatic Geometry with Algorithms

## Overview
This phase merges the geometric advantages of bistatic radar with the processing advantages of advanced enhancement algorithms.

The purpose is to evaluate how both approaches interact together in a unified radar framework.

This becomes the core experimental stage of the entire project.

---

## Main Objectives

- Integrate enhancement algorithms into bistatic simulations
- Measure SNR improvement
- Compare detection quality across configurations
- Generate enhanced range maps
- Analyze algorithm performance under varying geometry

---

## Combined Processing Pipeline

The new processing chain includes:

1. Bistatic signal generation
2. Geometry-aware delay modelling
3. Beat signal computation
4. FFT processing
5. Advanced enhancement algorithms
6. CFAR detection
7. Performance evaluation

This created a complete end-to-end enhanced radar simulation system.

---

## Experimental Analysis

Several experiments were conducted using:

- Different bistatic baselines
- Multiple target configurations
- Noise variation
- Weak target scenarios
- Different FFT sizes
- Alternative CFAR approaches

The goal was to identify which combinations produced the strongest improvement.

---

## SNR Measurement

Signal-to-noise ratio improvements were measured before and after algorithm application.

Metrics studied:

- Detection confidence
- Peak sharpness
- Noise floor reduction
- False alarm behaviour

This stage quantitatively demonstrated the impact of enhancement methods.

---

## Visualization Work

Different plots and visual outputs were generated:

- Enhanced range maps
- Range-Doppler comparisons
- SNR improvement graphs
- Detection overlays
- Baseline comparison visuals

These visualizations became important for the final report and presentation.

---

## Key Outcomes

At the end of this phase:

- Bistatic and algorithmic enhancements were successfully integrated
- Performance gains became measurable
- Enhanced detection behaviour was observed
- The complete experimental framework became ready for evaluation

---

# Phase 5 — Evaluation & Results

## Overview
The final phase focuses on performance evaluation, benchmarking, and result interpretation.

This stage transforms the project from a simulation prototype into a research-oriented study with measurable outcomes.

The emphasis is placed on comparing approaches and validating whether the proposed enhancements genuinely improve radar performance.

---

## Main Objectives

- Compare monostatic and bistatic radar performance
- Benchmark enhancement algorithms
- Measure detection probability
- Analyze SNR improvements
- Generate publication-style plots and results
- Summarize overall findings

---

## Comparative Analysis

The following comparisons were performed:

| Comparison | Purpose |
|---|---|
| Monostatic vs Bistatic | Geometry impact |
| Standard FFT vs Enhanced FFT | Resolution improvement |
| Basic CFAR vs Improved CFAR | Detection reliability |
| Non-integrated vs Integrated Signals | SNR improvement |

These comparisons helped identify the strongest contributors to performance gain.

---

## Detection Probability Analysis

Detection probability experiments were conducted under varying:

- Noise conditions
- Target ranges
- Reflection strengths
- Baseline distances
- Algorithm selections

This helped evaluate real-world robustness.

---

## Performance Metrics

The following metrics were analyzed:

- Range resolution
- Detection probability
- False alarm rate
- SNR improvement
- Peak localization accuracy
- Weak target detection capability

These metrics formed the basis of the final conclusions.

---

## Final Outputs

The final deliverables include:

- Complete FMCW radar simulation
- Bistatic geometry framework
- Range enhancement algorithms
- Evaluation plots
- Comparative analysis report
- Final presentation material

---

## Final Conclusion

The project demonstrates that combining bistatic radar geometry with modern signal processing algorithms can significantly improve radar detection capability and range estimation quality.

Instead of relying on a single enhancement method, the project explores how geometric diversity and algorithmic intelligence complement each other to create a more capable radar sensing system.

The final system serves as both:

- A research-oriented simulation framework
- A learning platform for advanced FMCW radar concepts

---

# Final Output — Simulation + Range Enhancement Report

## Project Summary

This project develops a complete FMCW radar simulation framework capable of:

- Monostatic radar processing
- Bistatic geometry modelling
- Range enhancement techniques
- Advanced detection algorithms
- Comparative performance evaluation

The workflow progresses from basic radar understanding to a fully enhanced and experimentally validated radar processing pipeline.

---

## Overall Highlights

- End-to-end FMCW radar simulation
- Geometry-aware bistatic modelling
- Advanced FFT and CFAR implementations
- Super-resolution experimentation
- SNR improvement analysis
- Comprehensive visualization pipeline
- Research-oriented evaluation framework

---

## Future Scope

Possible future improvements include:

- Real hardware integration
- MIMO radar expansion
- Deep learning based target classification
- Real-time processing acceleration
- GPU-based signal processing
- Multi-target tracking systems
- Clutter modelling for outdoor environments

---

## Tools & Libraries

| Tool | Role |
|---|---|
| Python | Core development |
| MATLAB | Signal analysis |
| radarsimpy | Radar simulation |
| NumPy | Numerical computation |
| Matplotlib | Visualization |
| Jupyter Notebook | Experimentation |

---

## Author Notes

This project was designed not only as a simulation exercise but also as a deep exploration of how radar systems process information from raw electromagnetic reflections to meaningful target detections.

Each phase intentionally builds on the previous one, gradually moving from theory to implementation and finally toward measurable performance enhancement.

