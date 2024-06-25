# Sport video analysis for billiard matches
Computer vision project regarding sport video analysis for billiard matches
Useful functions to use:
1) boundingRect():Calculates the up-right bounding rectangle of a point set or non-zero pixels of gray-scale image.
https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga103fcbda2f540f3ef1c042d6a9b35ac7

2) boxPoints(): Finds the four vertices of a rotated rect. Useful to draw the rotated rectangle. 

3) In order to open and display videos using opencv it is required to ensure that OpenCV is built with the necessary codecs (like FFMPEG).

4) For each member of the team, in order to compile and run the code, we need to use CMake, so follow the following steps:
	- create in the root directory of the project, the "build" directory
	- put the CMakeLists.txt at the root directory
	- go to the build dir, and execute "cmake .."
	- then execute "make"
	- go back to the root directory and create a .gitignorefile
	- add in the .gitignorefile build/ in order to ignore the folder otherwise will be committed in the repository

