#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
//#include <QDebug>
#include <vector>


#include <iostream>

#include "LBF_api.h"

using namespace std;
using namespace cv;

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip);

int main()
{
	VideoCapture cap(0);    //打开默认摄像头
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	if (!cap.isOpened())
	{
		return -1;
	}
	Initial_Model("D:\\Work\\Projects\\multcamera\\FaceAlignment\\FaceAlignment\\model");
	Mat frame;
	Mat edges;

	CascadeClassifier cascade, nestedCascade;
	bool stop = false;
	//训练好的文件名称，放置在可执行文件同目录下
	//cascade.load("haarcascade_frontalface_default.xml");
	cascade.load("D:\\ork\\Projects\\multcamera\\FaceAlignment\\FaceAlignment\\test\\opencv-detection\\cascade.xml");
	nestedCascade.load("haarcascade_eye_tree_eyeglasses.xml");
	while (!stop)
	{
		cap >> frame;
		detectAndDraw(frame, cascade, nestedCascade, 2, 0);
		if (waitKey(33) >= 0)
			stop = true;
	}
	return 0;
}
void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip)
{
	int i = 0;
	double t = 0;
	//建立用于存放人脸的向量容器
	vector<Rect> faces, faces2;
	//定义一些颜色，用来标示不同的人脸
	const static Scalar colors[] = { CV_RGB(255,255,255) };
	//建立缩小的图片，加快检测速度
	//int cvRound (double value) 对一个double型的数进行四舍五入，并返回一个整型数！
	Mat gray;
	Mat smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);
	//转成灰度图像，Harr特征基于灰度图
	cvtColor(img, gray, CV_BGR2GRAY);
	//改变图像大小，使用双线性差值
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	//变换后的图像进行直方图均值化处理
	equalizeHist(smallImg, smallImg);

	//程序开始和结束插入此函数获取时间，经过计算求得算法执行时间
	t = (double)cvGetTickCount();

	cascade.detectMultiScale(smallImg, faces,
		1.1, 2, 0
		//|CV_HAAR_FIND_BIGGEST_OBJECT
		//|CV_HAAR_DO_ROUGH_SEARCH
		| CV_HAAR_SCALE_IMAGE
		,
		Size(20, 20));
	//如果使能，翻转图像继续检测
	if (tryflip)
	{
		flip(smallImg, smallImg, 1);
		cascade.detectMultiScale(smallImg, faces2, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));
		for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++)
		{
			faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
		}
	}
	t = (double)cvGetTickCount() - t;
	//   qDebug( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
	cout << "detected number: " << faces.size() << " " << "detection time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;
	for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		Mat smallImgROI;
		vector<Rect> nestedObjects;
		Point center;
		Scalar color = colors[0];
		int radius;

		double aspect_ratio = (double)r->width / r->height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			//标示人脸时在缩小之前的图像上标示，所以这里根据缩放比例换算回去
			rectangle(img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)), cvPoint(cvRound((r->x + r->width - 1)*scale), cvRound((r->y + r->height - 1)*scale)), color, 2, 8, 0);
		}
		else
		{
			center.x = cvRound((r->x + r->width*0.5)*scale);
			center.y = cvRound((r->y + r->height*0.5)*scale);
			radius = cvRound((r->width + r->height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		}

		smallImgROI = smallImg(*r);
		//检测关键点
		vector<Point2f> landmarks;
		detectKeyPoints(smallImgROI, landmarks);
		vector<Point2f>::iterator p_point = landmarks.begin();
		while (p_point != landmarks.end())
		{
			center.x = cvRound((r->x + p_point->x)*scale);
			center.y = cvRound((r->y + p_point->y)*scale);
			circle(img, center, 3, color, 1, 8, 0);
			p_point++;
		}
	}
	cv::imshow("result", img);
}
