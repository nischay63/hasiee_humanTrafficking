# hasiee_humanTrafficking


## Human Trafficking Problem

## Proposed Solutions
Since there are survelince camera installed in almost every important part of the city. It can be leverged to detect abnormal activities. We propose a Deep learning based solution which can analyze live feed of the camera in real time and further detect any kind of anamalous activity.

## What is anamalous activity
What is an anamalous activity?
Any activity which differs from a normal activity above a calculated threshold can be marked as anamalous activity. For example - A busy road in delhi can be considered as a normal activity whereas riots or voilent activites on the same roads will be considered anamalous.

## Challenge
Lack of publically available videos/data on rape, Human trafficking etc

## Solution
We propose to use Generative Adversar-ial Nets (GANs), which are trained using normal frames and corresponding optical-flow images in order to learn an internal representation of the scene normality. Since our GANs are trained with only normal data, they are not able to generate abnormal events.  At testing time the real data are compared with both the appearance and the motion representations re-constructed by our GANs and abnormal areas are detected by computing  local  differences.[reference]
