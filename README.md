# Finding Lane Lines on the Road
________________________________________
##### Finding Lane Lines on the Road
The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report
________________________________________
## Reflection
#### Description of the pipeline
###### _Pipeline can be found in either P1_lanes.py or P1.ipynb_

My pipeline consisted of 6 steps for processing the images. I converted the images to grayscale, used a Gaussian blur, applied a Canny transform, filtered out a region of interest, drew Hough lines, and added the single lines onto the original images.

Image after Canny Transform:

![image](https://user-images.githubusercontent.com/74683142/122586929-3ec1ee00-d02b-11eb-967e-1236038c3de6.png)

Final output:

![image](https://user-images.githubusercontent.com/74683142/122587018-58fbcc00-d02b-11eb-8216-0a314394780c.png)
 
In order to draw a single line on the left and right lanes, I modified the draw_lines() function.

* Iterated through the hough lines to get the slopes and y-intercepts, and separated the left and right lines by slope. 
* Used the average slope and y-intercept of each side and used those values to calculate two new points. 
* Extended both lines using the y_min and y_max of the region of interest and found the corresponding x-values to plot the single lines.

#### Potential shortcomings with the current pipeline
* One potential shortcoming would be applying this straight line fit to curved lanes. The straight line approximation was sufficient for the images in this case, but it did become a problem with the challenge video.
*	Another shortcoming could be noise within the lane, such as in the challenge video. There were color variations within the lane, as well as other noise such as shadows from the trees. This made it difficult to filter out the lane markings accurately, using the current pipeline.

#### Possible improvements to the pipeline
*	A possible improvement would be to add a higher order best fit line, to incorporate the change in curvature along the y-axis.
* Another potential improvement could be to filter the image based on color as well, to narrow it down to white or yellow lane markings. This could have helped to differentiate between changes in lane color or shadows from trees.
