# hasiee_humanTrafficking
Human Trafficking is one of the major concerns in the society. With the boom in Artificial Intelligence we propose a solution that exploits its high computational power to solve the issue.

## Proposed Solutions
 We propose a Deep learning based solution which can analyze live feed of the camera in real time and further detect any kind of anamalous activity. Details such as location of the incident can be sent to nearest police stations and hospitals

## What is an anamalous activity?
Any activity which differs from a normal activity above a calculated threshold can be marked as anamalous activity. For example - A busy road in delhi can be considered as a normal activity whereas riots or voilent activites on the same roads will be considered anamalous.

## Major Challenge
Lack of publically available videos/data on rape, human trafficking etc.

## Descriptive Solution
We propose to use Generative Adversarial Nets (GANs) which are trained using normal frames and corresponding optical-flow images in order to learn an internal representation of the scene normality. Since our GANs are trained with only normal data, they are not able to generate abnormal events.  At testing time the real data is compared with both the appearance and the motion representations reconstructed by our GANs and abnormal areas are detected by computing local differences.

## Team
Nischay Gupta
Nikhil Iyer
Saurabh Khandelwal

## Reference
[1] [ABNORMAL EVENT DETECTION IN VIDEOS USING GENERATIVE ADVERSARIAL NETS](https://arxiv.org/pdf/1708.09644.pdf)