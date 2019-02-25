Dataset

The dataset used will be USCD dataset for anomaly detection. 

The UCSD Anomaly Detection Dataset was acquired with a stationary camera mounted at an elevation, overlooking pedestrian walkways. The crowd density in the walkways was variable, ranging from sparse to very crowded. In the normal setting, the video contains only pedestrians. Abnormal events are due to either:

*the circulation of non-pedestrian entities in the walkways

*anomalous pedestrian motion patterns

*Hyperactivites life violence etc

Commonly occurring anomalies include bikers, skaters, small carts, and people walking across a walkway or in the grass that surrounds it. A few instances of people in wheelchair were also recorded. All abnormalities are naturally occurring, i.e. they were not staged for the purposes of assembling the dataset. The data was split into 2 subsets, each corresponding to a different scene. The video footage recorded from each scene was split into various clips of around 200 frames.

Peds1: clips of groups of people walking towards and away from the camera, and some amount of perspective distortion. Contains 34 training video samples and 36 testing video samples. 

Peds2: scenes with pedestrian movement parallel to the camera plane. Contains 16 training video samples and 12 testing video samples.

For each clip, the ground truth annotation includes a binary flag per frame, indicating whether an anomaly is present at that frame. In addition, a subset of 10 clips for Peds1 and 12 clips for Peds2 are provided with manually generated pixel-level binary masks, which identify the regions containing anomalies. This is intended to enable the evaluation of performance with respect to ability of algorithms to localize anomalies.

You can download the dataset [here](http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz)
