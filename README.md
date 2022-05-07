# Computer Vision

## Run the UI
```
python main.py
```

## 1 Apply Harris Operator

### There are 2 parameters need to be set for harris operator:
* Sensitivity: Sensitivity factor to separate corners from edges. (Small values result in detection of sharp corners).
* Threshold: Value used computing local maxima (Higher threshold means less corners)

### Harris corners with 0.05 Sensitivity and 0.01 threshold (using UI)

![harris ui](/images/harris_tab.png)

### Harris corners with 0.05 Sensitivity and 0.01 threshold  &#x2611;

Original image             |  Harris output
:-------------------------:|:-------------------------:
![harris input](/images/harris_image_400.jpg) |  ![harris output](/images/Harris_output.jpg)
![harris input](/images/cow_step_harris.png) |  ![harris output](/images/cow_Harris_output.jpg)

- Computation time of Harris is parely noticed = 0.02104 seconds to detect all the corners in the first image and 0.03479 seconds in the second one.


## 2 Apply SIFT

### SIFT with starting blurring level of 1.6 (using UI)

![sift ui](/images/sift_tab.png)


### SIFT with starting blurring level of 1.6   &#x2611;

Original image             |         SIFT output                   |  OPENCV output for SIFT
:-------------------------:|:------------------------------------:|:-------------------------:
![SIFT input](/images/cat.jpg) |  ![SIFT output](/images/sift_cat.jpeg) |  ![SIFT output](/images/sift_openCv.jpeg)

- Computation time of SIFT noticed = 42 seconds .
- ps : time may vary depending on CPU performance so this is average results of computation time


## 3 Match the image set features

### Normalized cross correlations (NCC) & sum of squared differences (SSD) (using UI)

![feature matching ui](/images/feature_tab.png)


### Using normalized cross correlations (NCC)   &#x2611;

![normalized cross correlations](/images/ncc.png)
Computation time of Normalized Cross Correlation =   3.942026138305664  seconds

### Using sum of squared differences (SSD)  &#x2611;

![time of Sum Square Distance](/images/ssd.png)
Computation time of Sum Square Distance =   14.870997190475464  seconds