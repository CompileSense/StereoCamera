//单目摄像头标定
//版本：Version 2.1.0
//先对摄像头进行单独标定，生成yml文件供双目标定使用
#include <opencv2/opencv.hpp>  
#include <highgui.hpp>  
#include "cv.h"  
#include <cv.hpp>  
#include <iostream> 
#include "SoloCalibration.h"
using namespace std;
using namespace cv;

const int imageWidth = 640;                             //摄像头的分辨率  
const int imageHeight = 480;
const int boardWidth = 7;                               //横向的角点数目  
const int boardHeight = 5;                              //纵向的角点数据  
const int boardCorner = boardWidth * boardHeight;       //总的角点数据  
const int frameNumber = 10;                             //相机标定时需要采用的图像帧数  
const int squareSize = 35;                              //标定板黑白格子的大小 单位mm  
const Size boardSize = Size(boardWidth, boardHeight);   
Mat intrinsic;                                          //相机内参数  
Mat distortion_coeff;                                   //相机畸变参数  
vector<Mat> rvecs;                                        //旋转向量  
vector<Mat> tvecs;                                        //平移向量  
vector<vector<Point2f>> corners;                        //各个图像找到的角点的集合 和objRealPoint 一一对应  
vector<vector<Point3f>> objsoloRealPoint;                   //各副图像的角点的实际物理坐标集合  


vector<Point2f> corner;                                   //某一副图像找到的角点  



														  //计算标定板上模块的实际物理坐标
void calRealsoloPoint(vector<vector<Point3f>>& obj, int boardwidth, int boardheight, int imgNumber, int squaresize)
{
	//  Mat imgpoint(boardheight, boardwidth, CV_32FC3,Scalar(0,0,0));  
	vector<Point3f> imgpoint;
	for (int rowIndex = 0; rowIndex < boardheight; rowIndex++)
	{
		for (int colIndex = 0; colIndex < boardwidth; colIndex++)
		{
			//  imgpoint.at<Vec3f>(rowIndex, colIndex) = Vec3f(rowIndex * squaresize, colIndex*squaresize, 0);  
			imgpoint.push_back(Point3f(rowIndex * squaresize, colIndex * squaresize, 0));
		}
	}
	for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
	{
		obj.push_back(imgpoint);
	}
}

//设置相机的初始参数 也可以不估计
void guessCameraParam(void)
{
	/*分配内存*/
	intrinsic.create(3, 3, CV_64FC1);
	distortion_coeff.create(5, 1, CV_64FC1);

	/*
	fx 0 cx
	0 fy cy
	0 0  1
	*/
	intrinsic.at<double>(0, 0) = 256.8093262;   //fx         
	intrinsic.at<double>(0, 2) = 160.2826538;   //cx  
	intrinsic.at<double>(1, 1) = 254.7511139;   //fy  
	intrinsic.at<double>(1, 2) = 127.6264572;   //cy  

	intrinsic.at<double>(0, 1) = 0;
	intrinsic.at<double>(1, 0) = 0;
	intrinsic.at<double>(2, 0) = 0;
	intrinsic.at<double>(2, 1) = 0;
	intrinsic.at<double>(2, 2) = 1;

	/*
	k1 k2 p1 p2 p3
	*/
	distortion_coeff.at<double>(0, 0) = -0.193740;  //k1  
	distortion_coeff.at<double>(1, 0) = -0.378588;  //k2  
	distortion_coeff.at<double>(2, 0) = 0.028980;   //p1  
	distortion_coeff.at<double>(3, 0) = 0.008136;   //p2  
	distortion_coeff.at<double>(4, 0) = 0;          //p3  
}

void outputSoloParam(int num)
{
	/*保存数据*/
	//cvSave("cameraMatrix.xml", &intrinsic);  
	//cvSave("cameraDistoration.xml", &distortion_coeff);  
	//cvSave("rotatoVector.xml", &rvecs);  
	//cvSave("translationVector.xml", &tvecs);  
	/*输出数据*/
	cout << "fx :" << intrinsic.at<double>(0, 0) << endl << "fy :" << intrinsic.at<double>(1, 1) << endl;
	cout << "cx :" << intrinsic.at<double>(0, 2) << endl << "cy :" << intrinsic.at<double>(1, 2) << endl;
	cout << "k1 :" << distortion_coeff.at<double>(0, 0) << endl;
	cout << "k2 :" << distortion_coeff.at<double>(1, 0) << endl;
	cout << "p1 :" << distortion_coeff.at<double>(2, 0) << endl;
	cout << "p2 :" << distortion_coeff.at<double>(3, 0) << endl;
	cout << "p3 :" << distortion_coeff.at<double>(4, 0) << endl;
	
	if (num == 0)
	{
		FileStorage fs("..\\StereoCamera\\data\\CameraIntrinsicsR.yml", FileStorage::WRITE);
		if (fs.isOpened())
		{   
			fs << "cameraMatrixR" << intrinsic << "distCoeffR" << distortion_coeff;
			fs.release();
			cout << "Matrix Right write successful!" << endl;
		}
		else
			cout << "Error: can not save the extrinsic parameters\n";
	}
	if (num == 1)
	{
		FileStorage fs("..\\StereoCamera\\data\\CameraIntrinsicsL.yml", FileStorage::WRITE);
		if (fs.isOpened())
		{
			fs << "cameraMatrixL" << intrinsic << "distCoeffL" << distortion_coeff;
			fs.release();
			cout << "Matrix Left write successful!" << endl;
		}
		else
			cout << "Error: can not save the extrinsic parameters\n";
	}
	
}


int SoloCalibration()
{
	int Capturenum ;
	for (Capturenum = 0;Capturenum < 2;Capturenum++)
	{
		//打开摄像头
		VideoCapture capture(Capturenum);
		// 检查摄像头打开是否成功
		if (!capture.isOpened())
		{
			return -1;
			cout << "Fail to open the capture!" << endl;
		}
		cout << "Press W to analysis correct frame." << endl << "Press Q to quit the program." << endl;

		int goodFrameCount = 0;
		Mat img, grayImage;

		//取得9张适合检测的棋盘图片
		while (goodFrameCount < frameNumber)
		{
			//从摄像头取得一帧
			capture >> img;

			//resize(img, img, img.size()/2, 0, 0, INTER_AREA);
			//显示一帧画面
 			imshow("Capture", img);

			//读取按键W
			if (cvWaitKey(10) == 'w')
			{
				Mat rgbImage = img;
				//检查是否能够找到脚点
				bool isFind = findChessboardCorners(rgbImage, boardSize, corner, 0);
				//所有角点都被找到 说明这幅图像是适合检测的
				if (isFind == true)
				{
					/*
					Size(5,5) 搜索窗口的一半大小
					Size(-1,-1) 死区的一半尺寸
					TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1)迭代终止条件
					*/
					cvtColor(img, grayImage, CV_BGR2GRAY);
					cornerSubPix(grayImage, corner, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
					//在图像上画出角点
					drawChessboardCorners(rgbImage, boardSize, corner, isFind);
					//显示画点后的图像
					imshow("chessboard", rgbImage);
					//将角点信息存入corner
					corners.push_back(corner);

					goodFrameCount++;
					cout << "The image No." << goodFrameCount << " is good!" << endl;
				}
				//如果图像不符合检测标准
				else
				{
					cout << "The image No." << goodFrameCount + 1 << " is bad please try again!" << endl;
				}
			}
			//读取按键Q
			if (cvWaitKey(10) == 'q')
				return 0;
		}

		//图像采集完毕 接下来开始摄像头的校正
		//calibrateCamera()
		//输入参数 objectPoints  角点的实际物理坐标
		//imagePoints   角点的图像坐标
		//imageSize     图像的大小
		//输出参数
		//cameraMatrix  相机的内参矩阵
		//distCoeffs    相机的畸变参数
		//rvecs         旋转矢量(外参数)
		//tvecs         平移矢量(外参数）


		/*设置实际初始参数 根据calibrateCamera来 如果flag = 0 也可以不进行设置*/
		guessCameraParam();
		cout << "guess successful" << endl;
		/*计算实际的校正点的三维坐标*/
		calRealsoloPoint(objsoloRealPoint, boardWidth, boardHeight, frameNumber, squareSize);
		cout << "cal real successful" << endl;
		/*标定摄像头*/
		calibrateCamera(objsoloRealPoint, corners, Size(imageWidth, imageHeight), intrinsic, distortion_coeff, rvecs, tvecs, 0);
		cout << "calibration successful" << endl;
		/*保存并输出参数*/
		outputSoloParam(Capturenum);
		cout << "out successful" << endl;

		/*显示畸变校正效果*/
		Mat cImage;
		undistort(img, cImage, intrinsic, distortion_coeff);
		imshow("Corret Image", cImage);
		cout << "Wait for Key" << endl;
		waitKey(0);
		destroyWindow("Corret Image");
	}
	destroyWindow("chessboard");
	destroyWindow("Capture");
	return 0;
}