# Apple Tracking Project/ Apple Harvesting Project
### Carnegie Mellon University & UC Davis
## Point Cloud
### Pre-requisites:
OpenCV Version: 4.5.4 </br>
PCL Version: 1.7.3 (pcl_ros) </br>
g++ Version: 9.3.0 </br>
Note: </br>
Use ```pkg-config --modversion PKGNAME``` to check versions </br>
Use ```pkg-config --list-all | grep PKGNAME``` to see if your library was correctly installed, or might have to add additional ```PKG_CONFIG_PATH``` to ```.bashrc``` in the following manner: </br>
1.) Open ```.bashrc``` </br>
2.) Insert ```PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig``` (or wherever you store your packages) and ```export PKG_CONFIG_PATH``` </br>
3.) Save & close ```.bashrc```
### Codes:
```pcl_process.cpp``` - Processes pairs of **RECTIFIED** stereo images using OpenCV's SGBM algorithm to obtain disparity map, generates point clouds from disparity map, filters each point cloud using Statistical Outlier Removal (SOR) filtering, and saves the processed point cloud into a folder. To compile, type ```./pcl_process_compile.bat``` and execute program by typing ```./output```. </br>
### Outputs:
Source Image (Left image shown):
![L0007](https://user-images.githubusercontent.com/71652695/137849378-029496d9-006f-499d-8288-d9ab3b0a60bd.jpeg)
Disparity Map
![L0007_disp](https://user-images.githubusercontent.com/71652695/137849342-891759cb-59f1-4167-9492-800adde15195.png)
Point Cloud
![L0007_pc](https://user-images.githubusercontent.com/71652695/137849768-f7c33bb3-2aa2-4ecd-9419-51d95079052e.png)
