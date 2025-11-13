# Planning File


## November 5th, 2025
Want to access the comma ai dataset, and then analyze the video frame by frame, overlaying my ai predictions onto each part.


## August 17th, 2025
Attempting to fix the pth file

* the pth file was totally okay, it was just saving as a dictionary of dictionaries which I did not realize. I thought it was just a single dictionary.
* However the visualization is still completely off for some reason.


## August 16th, 2025
Implemented residual neural networks, and fixed the folder creation system.

Actually running the neural network isn't working since the pth file might be improperly constructed.


## August 8th, 2025
Today I am going to attempt residual networks.

## August 6th, 2025
I'm decently sure that all this time my pth tests were actually fine. The real problem was that my neural network was giving the same lane prediction for every image because likely it isn't complex enough. 

Improvements for the future:
 - More complex neural network
 - Clean up downsize.py
 - Clean up pth_test.py, delete pth_tests.py
 - make masterfile.py copy itself to each saved model -> done
 - separate nn model into it's own file -> done
 - Batch normalization
 - Residual neural networks


## August 3rd, 2025
Goals:
 - Achieve functioning pth_test


## August 1st, 2025
Goals:
 - Visualize Neural Network overlayed on the image
 - Begin trying augmented reality
 - Start learning OpenStreetMaps
