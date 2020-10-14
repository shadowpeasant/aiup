# Using LabelImg to annotate images

### Exercise

1. In local computer command prompt type labelImg, the labelImage application will launch.

![image](https://www.linkpicture.com/q/labelimage_1.jpg)






2. Use the internet browser download a image(jpg) eg. image.jpg with cars insides.

![image](https://www.linkpicture.com/q/images_141.jpg)





3. In labelImage application,select menu Open. Then look for the downloaded image file name eg. image.jpg. Select file and click Open.

![image](https://www.linkpicture.com/q/labelimage_2.jpg)


4. Once the image is loaded. Goto menu Edit select Create RectBox. Use the cursor to draw a box to house the car inside.

![image](https://www.linkpicture.com/q/labelimage_3.jpg)


5. Enter the label name for the object in the box and click on OK button. In the case, our object will car(label).

![image](https://www.linkpicture.com/q/labelimage_4.jpg)

6. Repeat the step 5 and 6 to complete the annotation of the other cars in the image.

![image](https://www.linkpicture.com/q/labelimage_5.jpg)

7. Select menu File and set the annotation output to be PascalVOC. It is the format for the label and bounding box parameters to be stored in the XML file.

![image](https://www.linkpicture.com/q/labelimage_6.jpg)


8. Select menu File and choose Save. 

![image](https://www.linkpicture.com/q/labelimage_7.jpg)


9.  At the dialog box, you will see a XML file name with the same name as your image file. Click on Save button, a XML file will save into the same directory where your image is.

![image](https://www.linkpicture.com/q/labelimage_8.jpg)


Open up the xml file with any text editor to observe the content inside it.

Can you find where is the label and annotation parameters for each of the car?
<details><summary>Click here for answer</summary> 
<br/>
    The car label is stored inside the XML tag name 
    
    ```
    
    <name>car</name>
    
    
    ```

    The bounding box is stored inside the XML tag name 
    
    ```
    <bndbox>
			<xmin>36</xmin>
			<ymin>22</ymin>
			<xmax>189</xmax>
			<ymax>118</ymax>
	</bndbox>

    ```

<br/>
</details>

This exercise let us have a feel how to do the image annotation to prepare data for object detection training.It is a tedious process to do it for all the images. Due time constraint we will use the dataset that is copied in the VM to do the rest of the lab.

