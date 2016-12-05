//双目匹配
//版本:Version 3.1.1
//利用双目标定文件对双目摄像头所采得的画面进行匹配并能够输出图像世界坐标
//加入特征点提取项目
//备注：需自行添加模型库和训练结果

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <vector>
#include <stdio.h>
#include <iostream>

#include "LBF_api.h"

using namespace cv;
using namespace std;

static void saveXYZ(const char* filename, const Mat& mat)
{
	const double max_z =80;
	FILE* fp = fopen(filename, "wt");
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}

void detectAndDraw(Mat& inputimg,Mat& dirimg,CascadeClassifier& cascade,
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
	Mat smallImg(cvRound(inputimg.rows / scale), cvRound(inputimg.cols / scale), CV_8UC1);
	//转成灰度图像，Harr特征基于灰度图
	cvtColor(inputimg, gray, CV_BGR2GRAY);
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
			rectangle(inputimg, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)), cvPoint(cvRound((r->x + r->width - 1)*scale), cvRound((r->y + r->height - 1)*scale)), color, 2, 8, 0);
		}
		else
		{
			center.x = cvRound((r->x + r->width*0.5)*scale);
			center.y = cvRound((r->y + r->height*0.5)*scale);
			radius = cvRound((r->width + r->height)*0.25*scale);
			circle(inputimg, center, radius, color, 3, 8, 0);
		}

		smallImgROI = smallImg(*r);
		//检测关键点
		vector<Point2f> landmarks;
		detectKeyPoints(smallImgROI, landmarks);
		vector<Point2f>::iterator p_point = landmarks.begin();
		//cout << landmarks << endl;
		while (p_point != landmarks.end())
		{
			center.x = cvRound((r->x + p_point->x)*scale);
			center.y = cvRound((r->y + p_point->y)*scale);
			circle(dirimg, center, 3, color, 1, 8, 0);
			p_point++;
		}
	}
	cv::imshow("result", dirimg);
}

int main()
{
	//读取yml双目匹配文件
	const char* intrinsic_filename = "data\\intrinsics.yml";
	const char* extrinsic_filename = "data\\extrinsics.yml";
	Rect roi1, roi2;
	Mat Q;
	float scale = 0.5;
	
	CascadeClassifier cascade, nestedCascade;
	Initial_Model("D:\\Work\\Projects\\multcamera\\FaceAlignment\\FaceAlignment\\model\\");
	cascade.load("D:\\Work\\Projects\\multcamera\\FaceAlignment\\FaceAlignment\\test\\opencv-detection\\haarcascade_frontalface_alt.xml");
	nestedCascade.load("haarcascade_eye_tree_eyeglasses.xml");
	
	// 从data文件中读取标定参数
	FileStorage fs(intrinsic_filename, FileStorage::READ);
	Mat M1, D1, M2, D2;
	fs["cameraMatrixL"] >> M1;
	fs["cameraDistcoeffL"] >> D1;
	fs["cameraMatrixR"] >> M2;
	fs["cameraDistcoeffR"] >> D2;
	M1 *= scale;
	M2 *= scale;

	fs.open(extrinsic_filename, FileStorage::READ);
	Mat R, T, R1, P1, R2, P2;
	fs["R"] >> R;
	fs["T"] >> T;

	//绘制窗口
	Mat canvas;
	int w, h;
	w = 320;
	h = 240;
	canvas.create(h, w * 2, CV_8UC3);

	Mat disp, disp8, img1, img2;
	//设置匹配模式:STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2, STEREO_VAR = 3
	int alg = 1;

	//匹配参数
	int SADWindowSize = 1, numberOfDisparities =128;
	////////////////////5////////////////////////256/
	bool no_display = false;


	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);

	Mat left, right;
	//打开左摄像头
	VideoCapture capleft(1);
	//打开右摄像头
	VideoCapture capright(0);
	
	cout << "Press Q to quit the program" << endl;

	while (1)
	{
		capleft >> left;
		capright >> right;
		//左上图像画到画布上
		//得到画布的一部分
		
		Mat canvasPart = canvas(Rect(0, 0, w, h));
		//把图像缩放到跟canvasPart一样大小
		resize(left, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);

		//右上图像画到画布上
		//获得画布的另一部分
		canvasPart = canvas(Rect(w, 0, w, h));
		resize(right, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);

		imshow("Capture", canvas);

		resize(left, img1, left.size() / 2, 0, 0, INTER_AREA);
		resize(right, img2, right.size() / 2, 0, 0, INTER_LINEAR);
		//img1 = left;
		//img2 = right;
		Size img_size = img1.size();

		//对两个摄像头所采得的画面进行矫正
		stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

		Mat map11, map12, map21, map22;
		initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
		initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

		Mat img1r, img2r;
		remap(img1, img1r, map11, map12, INTER_LINEAR);
		remap(img2, img2r, map21, map22, INTER_LINEAR);

		img1 = img1r;
		img2 = img2r;


		numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width / 8) + 15) & -16;

		sgbm->setPreFilterCap(63);
		int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
		sgbm->setBlockSize(sgbmWinSize);

		int cn = img1.channels();

		sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
		sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
		sgbm->setMinDisparity(0);
		sgbm->setNumDisparities(numberOfDisparities);
		sgbm->setUniquenessRatio(10);
		sgbm->setSpeckleWindowSize(100);
		sgbm->setSpeckleRange(32);
		sgbm->setDisp12MaxDiff(1);
		sgbm->setMode(StereoSGBM::MODE_SGBM);


		//显示运行时间
		int64 t = getTickCount();
		sgbm->compute(img1, img2, disp);
		//medianBlur(disp, disp, 9);
		t = getTickCount() - t;
		cout << "Time elapsed: " << t * 1000 / getTickFrequency() << "ms" << endl;

		//disp = dispp.colRange(numberOfDisparities, img1p.cols);
		disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));
		resize(disp8, disp8, canvasPart.size() * 2, 0, 0, INTER_LINEAR);
		imshow("Deepwindow", disp8);
		detectAndDraw(left,disp8, cascade, nestedCascade, 2, 0);
		if (cvWaitKey(10) == 'w')
		{

			printf("storing the point cloud...");
			Mat xyz;
			reprojectImageTo3D(disp, xyz, Q, true);
			saveXYZ("pointcloud.txt", xyz);
			printf("\n");
		};
		if (cvWaitKey(10) == 'q')
			break;
	}
	return 0;
}