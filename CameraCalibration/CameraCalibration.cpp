//双目摄像头立体标定
//版本：Version 2.1.0
//先对两个摄像头进行单独标定，然后在进行立体标定并生成相应yml文件
#include <opencv2/opencv.hpp>  
#include <highgui.hpp>  
#include "cv.h"  
#include <cv.hpp>  
#include <iostream>  
#include"SoloCalibration.h"
using namespace std;
using namespace cv;

const int imageWidth = 640;                             //摄像头的分辨率  
const int imageHeight = 480;
const int boardWidth = 7;                               //横向的角点数目  
const int boardHeight = 5;                              //纵向的角点数据  
const int boardCorner = boardWidth * boardHeight;       //总的角点数据  
const int frameNumber = 14;                             //相机标定时需要采用的图像张数  
const int squareSize = 35;                              //标定板黑白格子的大小 单位mm  
const Size boardSize = Size(boardWidth, boardHeight);   //  
Size imageSize = Size(imageWidth, imageHeight);

Mat R, T, E, F;                                         //R 旋转矢量 T平移矢量 E本征矩阵 F基础矩阵  
//vector<Mat> rvecs;                                        //旋转向量  
//vector<Mat> tvecs;                                        //平移向量  
vector<vector<Point2f>> imagePointL;                    //左边摄像机所有照片角点的坐标集合  
vector<vector<Point2f>> imagePointR;                    //右边摄像机所有照片角点的坐标集合  
vector<vector<Point3f>> objRealPoint;                   //各副图像的角点的实际物理坐标集合  


vector<Point2f> cornerL;                              //左边摄像机某一照片角点坐标集合  
vector<Point2f> cornerR;                              //右边摄像机某一照片角点坐标集合  

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;

Mat Rl, Rr, Pl, Pr, Q;                                  //校正旋转矩阵R，投影矩阵P 重投影矩阵Q (下面有具体的含义解释）   
Mat mapLx, mapLy, mapRx, mapRy;                         //映射表  
Rect validROIL, validROIR;                              //图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  

//相机内参矩阵
Mat cameraMatrixL;
Mat distCoeffL;
Mat cameraMatrixR ;
Mat distCoeffR ;

//读取标定矩阵函数
void ReadCameraparam(void)
{
	FileStorage as("..\\StereoCamera\\data\\CameraIntrinsicsL.yml", FileStorage::READ);
	as["cameraMatrixL"] >> cameraMatrixL;
	as["distCoeffL"] >> distCoeffL;
	FileStorage bs("..\\StereoCamera\\data\\CameraIntrinsicsR.yml", FileStorage::READ);
	bs["cameraMatrixR"] >> cameraMatrixR;
	bs["distCoeffR"] >> distCoeffR;
}

/*计算标定板上模块的实际物理坐标*/
void calRealPoint(vector<vector<Point3f>>& obj, int boardwidth, int boardheight, int imgNumber, int squaresize)
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

void outputCameraParam(void)
{
	/*保存数据*/
	/*输出数据*/
	FileStorage fs("..\\StereoCamera\\data\\intrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "cameraMatrixL" << cameraMatrixL << "cameraDistcoeffL" << distCoeffL << "cameraMatrixR" << cameraMatrixR << "cameraDistcoeffR" << distCoeffR;
		fs.release();
		cout << "cameraMatrixL=:" << cameraMatrixL << endl << "cameraDistcoeffL=:" << distCoeffL << endl << "cameraMatrixR=:" << cameraMatrixR << endl << "cameraDistcoeffR=:" << distCoeffR << endl;
	}
	else
	{
		cout << "Error: can not save the intrinsics!!!!!" << endl;
	}

	fs.open("..\\StereoCamera\\data\\extrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "Rl" << Rl << "Rr" << Rr << "Pl" << Pl << "Pr" << Pr << "Q" << Q;
		cout << "R=" << R << endl << "T=" << T << endl << "Rl=" << Rl << endl << "Rr=" << Rr << endl << "Pl=" << Pl << endl << "Pr=" << Pr << endl << "Q=" << Q << endl;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";
}

int main()
{
	//分别调试左右摄像头
	//SoloCalibration();
	ReadCameraparam();
	bool isFindL, isFindR;

	Mat canvas;
	int w, h;
	w = 320;
	h = 240;
	canvas.create(h * 3, w * 2, CV_8UC3);
	Mat left, right;

	int goodFrameCount = 0;
	// 打开左摄像头
	VideoCapture capleft(1);
	//打开右摄像头

	VideoCapture capright(0);
	// 检查左摄像头打开是否成功
	if (!capleft.isOpened())
		return -1;
	// 检查右摄像头打开是否成功
	if (!capright.isOpened())
		return -1;

	cout << "Press W to analysis correct frame." << endl << "Press Q to quit the program" << endl;

	while (goodFrameCount < frameNumber)
	{
		// 取得左摄像头一帧的画面并显示
		capleft >> left;
		// 取得右摄像头一帧的画面并显示
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
		//读取按键W
		if (cvWaitKey(10) == 'w')
		{
			//读取左边的图像
			rgbImageL = left;
			cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);

			//读取右边的图像
			rgbImageR = right;
			cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);
			//分别在两幅图片中寻找脚点
			isFindL = findChessboardCorners(rgbImageL, boardSize, cornerL);
			isFindR = findChessboardCorners(rgbImageR, boardSize, cornerR);


			//如果两幅图像都找到了所有的角点 则说明这两幅图像是可行的
			if (isFindL == true && isFindR == true)
			{
				/*
				Size(5,5) 搜索窗口的一半大小
				Size(-1,-1) 死区的一半尺寸
				TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1)迭代终止条件
				*/

				//在左图上画出脚点并显示
				cornerSubPix(grayImageL, cornerL, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
				drawChessboardCorners(rgbImageL, boardSize, cornerL, isFindL);
				//imshow("chessboardL", rgbImageL);
				//左中图像画到画布上
				canvasPart = canvas(Rect(0, h, w, h));
				resize(rgbImageL, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);

				//将左图脚点信息保存
				imagePointL.push_back(cornerL);

				//在右图上画出脚点并显示
				cornerSubPix(grayImageR, cornerR, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
				drawChessboardCorners(rgbImageR, boardSize, cornerR, isFindR);
				//imshow("chessboardR", rgbImageR);
				//右中图像画到画布上
				canvasPart = canvas(Rect(w, h, w, h));
				resize(rgbImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);

				//将右图脚点信息保存
				imagePointR.push_back(cornerR);

				goodFrameCount++;
				cout << "The image No." << goodFrameCount << " is good!" << endl;
			}

			else
			{
				cout << "The image No." << goodFrameCount + 1 << " is bad please try again!" << endl;
			}
		}
		putText(canvas, "LeftCapture", cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
		putText(canvas, "RightCapture", cv::Point(325, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
		putText(canvas, "LeftChessboard", cv::Point(5, 255), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
		putText(canvas, "RightChessboard", cv::Point(325, 255), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
		putText(canvas, "LeftRectified", cv::Point(5, 495), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
		putText(canvas, "RightRectified", cv::Point(325, 495), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
		imshow("Output", canvas);
		//读取按键Q
		if (waitKey(10) == 'q')
		{
			return 0;
		}
	}


	//计算实际的校正点的三维坐标
	//根据实际标定格子的大小来设置
	calRealPoint(objRealPoint, boardWidth, boardHeight, frameNumber, squareSize);
	cout << "cal real successful" << endl;


	//标定摄像头
	//由于左右摄像机分别都经过了单目标定
	//所以在此处选择flag = CALIB_USE_INTRINSIC_GUESS
	double rms = stereoCalibrate(objRealPoint, imagePointL, imagePointR,
		cameraMatrixL, distCoeffL,
		cameraMatrixR, distCoeffR,
		Size(imageWidth, imageHeight), R, T, E, F,
		CALIB_USE_INTRINSIC_GUESS,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));

	cout << "Stereo Calibration done with RMS error = " << rms << endl;


	//立体校正的时候需要两幅图像共面并且行对准 以使得立体匹配更加的可靠
	//使得两幅图像共面的方法就是把两个摄像头的图像投影到一个公共成像面上，这样每幅图像从本图像平面投影到公共图像平面都需要一个旋转矩阵R
	//stereoRectify 这个函数计算的就是从图像平面投影都公共成像平面的旋转矩阵Rl,Rr。 Rl,Rr即为左右相机平面行对准的校正旋转矩阵。
	//左相机经过Rl旋转，右相机经过Rr旋转之后，两幅图像就已经共面并且行对准了。
	//其中Pl,Pr为两个相机的投影矩阵，其作用是将3D点的坐标转换到图像的2D点的坐标:P*[X Y Z 1]' =[x y w]
	//Q矩阵为重投影矩阵，即矩阵Q可以把2维平面(图像平面)上的点投影到3维空间的点:Q*[x y d 1] = [X Y Z W]。其中d为左右两幅图像的时差

	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q,
		CALIB_ZERO_DISPARITY, -1, imageSize, &validROIL, &validROIR);

	//根据stereoRectify 计算出来的R 和 P 来计算图像的映射表 mapx,mapy
	//mapx,mapy这两个映射表接下来可以给remap()函数调用，来校正图像，使得两幅图像共面并且行对准
	//ininUndistortRectifyMap()的参数newCameraMatrix就是校正后的摄像机矩阵。在openCV里面，校正后的计算机矩阵Mrect是跟投影矩阵P一起返回的。
	//所以在这里传入投影矩阵P，此函数可以从投影矩阵P中读出校正后的摄像机矩阵

	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);


	Mat rectifyImageL, rectifyImageR;
	cvtColor(grayImageL, rectifyImageL, CV_GRAY2BGR);
	cvtColor(grayImageR, rectifyImageR, CV_GRAY2BGR);

	//imshow("Rectify Before", rectifyImageL);

	//remap后左右相机的图像共面并且行对准
	remap(rectifyImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
	remap(rectifyImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

	//imshow("ImageL", rectifyImageL);
	//imshow("ImageR", rectifyImageR);

	//左下图像画到画布上
	Mat canvasPart = canvas(Rect(0, h * 2, w, h));
	resize(rectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);

	//右下图像画到画布上
	canvasPart = canvas(Rect(w, h * 2, w, h));
	resize(rectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
	putText(canvas, "LeftRectified", cv::Point(5, 495), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
	putText(canvas, "RightRectified", cv::Point(325, 495), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
	imshow("Output", canvas);

	//输出矫正后图像
	//imwrite("ImageL",rectifyImageL);
	//imwrite("ImageR", rectifyImageR);


	/*保存并输出数据*/
	outputCameraParam();


	//把校正结果显示出来
	//把左右两幅图像显示到同一个画面上
	//这里只显示了最后一副图像的校正结果。并没有把所有的图像都显示出来

	Mat canvas_;
	double sf_;
	int w_, h_;
	sf_ = 600. / MAX(imageSize.width, imageSize.height);
	w_ = cvRound(imageSize.width * sf_);
	h_ = cvRound(imageSize.height * sf_);
	canvas_.create(h_, w_ * 2, CV_8UC3);

	//左图像画到画布上
	Mat canvasPart_ = canvas_(Rect(w_ * 0, 0, w_, h_));                                //得到画布的一部分  
	resize(rectifyImageL, canvasPart_, canvasPart_.size(), 0, 0, INTER_AREA);  //把图像缩放到跟canvasPart一样大小  
	Rect vroiL(cvRound(validROIL.x*sf_), cvRound(validROIL.y*sf_),                //获得被截取的区域    
		cvRound(validROIL.width*sf_), cvRound(validROIL.height*sf_));
	rectangle(canvasPart_, vroiL, Scalar(0, 0, 200), 3, 8);                      //画上一个矩形  

	cout << "Painted ImageL" << endl;

	//右图像画到画布上
	canvasPart_ = canvas_(Rect(w_, 0, w_, h_));                                      //获得画布的另一部分  
	resize(rectifyImageR, canvasPart_, canvasPart_.size(), 0, 0, INTER_LINEAR);
	Rect vroiR(cvRound(validROIR.x * sf_), cvRound(validROIR.y*sf_),
		cvRound(validROIR.width * sf_), cvRound(validROIR.height * sf_));
	rectangle(canvasPart_, vroiR, Scalar(0, 159, 0), 3, 8);

	cout << "Painted ImageR" << endl;

	//画上对应的线条
	for (int i = 0; i < canvas_.rows;i += 16)
		line(canvas_, Point(0, i), Point(canvas_.cols, i), Scalar(0, 200, 0), 1, 8);

	imshow("Rectified", canvas_);

	cout << "wait key" << endl;
	waitKey(0);
	system("pause");
	return 0;
}