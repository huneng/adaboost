# my_adaboost
## Introduce
This project is an inplementation of adaboost. I developed this project on ubuntu 14.04.
This project based on opencv-2.4.x which only used to decode image and some common image process just like resize.


## compile
Please be sure you in linux with opencv-2.4.x installed. If you want to work on other systems, 
you should rewrite the main function. Go into the source directory in terminal and type,
    make

there will generate three program in bin/ directory,
1) train    used to train model
2) detect   used to detect object
3) sample   used to generate trainning samples
4) main     no used

## run
I prepared some shell scripts in bin directory 
1) run.sh       tranning model scripts
2) detect.sh    detect object in image 

First, you shoud prepare trainning samples and list positive samples' path in file pos_list.txt and
list negative samples' path in file neg_list.txt. Then type 
    ./run.sh 

there will generate model file model.dat used to detect object. Then type 
    ./detect.sh [image]

to detect object in image. It will show the result and write to out.jpg

