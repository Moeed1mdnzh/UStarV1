### Dataset
The model is actually trained on a dataset made of a single 400x400 gray image.Here's how
we created 813 images of only one image(Figure 1) using image processing techniques:
1. A threshold is applied to the image to create a blank image of black and white
2. Next we use all available RGB colors to replace the white remaining pixel to create our X set. we use a step size to skip some colors and avoid repetition in our color distribution as well
3. We then only consider images from the X set to create labels with that have the sum larger than 45000000
4. In order to create our y set using the blank image, we use the below formula to create the labels: $$ {X \-(max \-img)} $$
   - X is the image created in step 3
   - max is the same image as our blank black and white image except the white region is replaced with the maximum value of the original image
   - img is the single image we used for the previous steps
5. The result of our label now has values below 0 and above 255 meaning that we should now replace any values below 0 with 0 and values above 255 with 255

![Preview of the data](https://github.com/Moeed1mdnzh/UStarV1/blob/master/assets/data.jpg)
<br />
***Preview of the data***
<br />



