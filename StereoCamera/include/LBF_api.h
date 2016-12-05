#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
// api

//模型参数初始化
void Initial_Model(string modelPath);
//单张人脸对齐
void single_FaceAlignment(string imgreadpath, string imgwritepath, vector<Point2f> &landmarks);
//多张人脸对齐
void patch_FaceAlignment(string imgfilelist, string imgwritelist);

void detectKeyPoints(Mat &img, vector<Point2f> &landmarks);
