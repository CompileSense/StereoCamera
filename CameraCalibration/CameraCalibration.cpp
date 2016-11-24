//˫Ŀ����ͷ����궨
//�汾��Version 2.0.1
//�ȶ���������ͷ���е����궨��Ȼ���ڽ�������궨��������Ӧyml�ļ�
#include <opencv2/opencv.hpp>  
#include <highgui.hpp>  
#include "cv.h"  
#include <cv.hpp>  
#include <iostream>  

using namespace std;
using namespace cv;

const int imageWidth = 640;                             //����ͷ�ķֱ���  
const int imageHeight = 480;
const int boardWidth = 7;                               //����Ľǵ���Ŀ  
const int boardHeight = 5;                              //����Ľǵ�����  
const int boardCorner = boardWidth * boardHeight;       //�ܵĽǵ�����  
const int frameNumber = 9;                             //����궨ʱ��Ҫ���õ�ͼ��֡��  
const int squareSize = 35;                              //�궨��ڰ׸��ӵĴ�С ��λmm  
const Size boardSize = Size(boardWidth, boardHeight);   //  
Size imageSize = Size(imageWidth, imageHeight);

Mat R, T, E, F;                                         //R ��תʸ�� Tƽ��ʸ�� E�������� F��������  
vector<Mat> rvecs;                                        //��ת����  
vector<Mat> tvecs;                                        //ƽ������  
vector<vector<Point2f>> imagePointL;                    //��������������Ƭ�ǵ�����꼯��  
vector<vector<Point2f>> imagePointR;                    //�ұ������������Ƭ�ǵ�����꼯��  
vector<vector<Point3f>> objRealPoint;                   //����ͼ��Ľǵ��ʵ���������꼯��  


vector<Point2f> cornerL;                              //��������ĳһ��Ƭ�ǵ����꼯��  
vector<Point2f> cornerR;                              //�ұ������ĳһ��Ƭ�ǵ����꼯��  

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;

Mat Rl, Rr, Pl, Pr, Q;                                  //У����ת����R��ͶӰ����P ��ͶӰ����Q (�����о���ĺ�����ͣ�   
Mat mapLx, mapLy, mapRx, mapRy;                         //ӳ���  
Rect validROIL, validROIR;                              //ͼ��У��֮�󣬻��ͼ����вü��������validROI����ָ�ü�֮�������  

														/*
														���ȱ궨�õ���������ڲξ���
														fx 0 cx
														0 fy cy
														0 0  1
														*/
Mat cameraMatrixL = (Mat_<double>(3, 3) <<
	820.046, 0, 234.353,
	0, 820.816, 234.353,
	0, 0, 1
	);
Mat distCoeffL = (Mat_<double>(5, 1) << -0.0175116, 1.2325, -0.000835355, -0.0050405, -8.22484);
/*
���ȱ궨�õ���������ڲξ���
fx 0 cx
0 fy cy
0 0  1

*/
Mat cameraMatrixR = (Mat_<double>(3, 3) <<
	814.123, 0, 306.948,
	0, 815.542, 231.294,
	0, 0, 1
	);
Mat distCoeffR = (Mat_<double>(5, 1) << -0.0267209, 2.15388, 0.00272984, 0.00485937, -14.9098);


/*����궨����ģ���ʵ����������*/
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
	/*��������*/
	/*�������*/
	FileStorage fs("../StereoCamera/data/intrinsics.yml", FileStorage::WRITE);
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

	fs.open("../StereoCamera/data/extrinsics.yml", FileStorage::WRITE);
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

	bool isFindL, isFindR;

	Mat canvas;
	int w, h;
	w = 320;
	h = 240;
	canvas.create(h * 3, w * 2, CV_8UC3);

	Mat left, right;

	int goodFrameCount = 0;
	// ��������ͷ
	VideoCapture capleft(1);
	//��������ͷ

	VideoCapture capright(0);
	// ���������ͷ���Ƿ�ɹ�
	if (!capleft.isOpened())
		return -1;
	// ���������ͷ���Ƿ�ɹ�
	if (!capright.isOpened())
		return -1;

	cout << "Press W to analysis correct frame." << endl << "Press Q to quit the program" << endl;

	while (goodFrameCount < frameNumber)
	{
		// ȡ��������ͷһ֡�Ļ��沢��ʾ
		capleft >> left;
		// ȡ��������ͷһ֡�Ļ��沢��ʾ
		capright >> right;

		//����ͼ�񻭵�������
		//�õ�������һ���� 
		Mat canvasPart = canvas(Rect(0, 0, w, h));
		//��ͼ�����ŵ���canvasPartһ����С
		resize(left, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);

		//����ͼ�񻭵�������
		//��û�������һ����
		canvasPart = canvas(Rect(w, 0, w, h));

		resize(right, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
		//��ȡ����W
		if (cvWaitKey(10) == 'w')
		{
			//��ȡ��ߵ�ͼ��
			rgbImageL = left;
			cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);

			//��ȡ�ұߵ�ͼ��
			rgbImageR = right;
			cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);
			//�ֱ�������ͼƬ��Ѱ�ҽŵ�
			isFindL = findChessboardCorners(rgbImageL, boardSize, cornerL);
			isFindR = findChessboardCorners(rgbImageR, boardSize, cornerR);


			//�������ͼ���ҵ������еĽǵ� ��˵��������ͼ���ǿ��е�
			if (isFindL == true && isFindR == true)
			{
				/*
				Size(5,5) �������ڵ�һ���С
				Size(-1,-1) ������һ��ߴ�
				TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1)������ֹ����
				*/

				//����ͼ�ϻ����ŵ㲢��ʾ
				cornerSubPix(grayImageL, cornerL, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
				drawChessboardCorners(rgbImageL, boardSize, cornerL, isFindL);
				//imshow("chessboardL", rgbImageL);
				//����ͼ�񻭵�������
				canvasPart = canvas(Rect(0, h, w, h));
				resize(rgbImageL, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);

				//����ͼ�ŵ���Ϣ����
				imagePointL.push_back(cornerL);

				//����ͼ�ϻ����ŵ㲢��ʾ
				cornerSubPix(grayImageR, cornerR, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
				drawChessboardCorners(rgbImageR, boardSize, cornerR, isFindR);
				//imshow("chessboardR", rgbImageR);
				//����ͼ�񻭵�������
				canvasPart = canvas(Rect(w, h, w, h));
				resize(rgbImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);

				//����ͼ�ŵ���Ϣ����
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
		//��ȡ����Q
		if (waitKey(10) == 'q')
		{
			return 0;
		}
	}


	//����ʵ�ʵ�У�������ά����
	//����ʵ�ʱ궨���ӵĴ�С������
	calRealPoint(objRealPoint, boardWidth, boardHeight, frameNumber, squareSize);
	cout << "cal real successful" << endl;


	//�궨����ͷ
	//��������������ֱ𶼾����˵�Ŀ�궨
	//�����ڴ˴�ѡ��flag = CALIB_USE_INTRINSIC_GUESS
	double rms = stereoCalibrate(objRealPoint, imagePointL, imagePointR,
		cameraMatrixL, distCoeffL,
		cameraMatrixR, distCoeffR,
		Size(imageWidth, imageHeight), R, T, E, F,
		CALIB_USE_INTRINSIC_GUESS,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));

	cout << "Stereo Calibration done with RMS error = " << rms << endl;


	//����У����ʱ����Ҫ����ͼ���沢���ж�׼ ��ʹ������ƥ����ӵĿɿ�
	//ʹ������ͼ����ķ������ǰ���������ͷ��ͼ��ͶӰ��һ�������������ϣ�����ÿ��ͼ��ӱ�ͼ��ƽ��ͶӰ������ͼ��ƽ�涼��Ҫһ����ת����R
	//stereoRectify �����������ľ��Ǵ�ͼ��ƽ��ͶӰ����������ƽ�����ת����Rl,Rr�� Rl,Rr��Ϊ�������ƽ���ж�׼��У����ת����
	//���������Rl��ת�����������Rr��ת֮������ͼ����Ѿ����沢���ж�׼�ˡ�
	//����Pl,PrΪ���������ͶӰ�����������ǽ�3D�������ת����ͼ���2D�������:P*[X Y Z 1]' =[x y w]
	//Q����Ϊ��ͶӰ���󣬼�����Q���԰�2άƽ��(ͼ��ƽ��)�ϵĵ�ͶӰ��3ά�ռ�ĵ�:Q*[x y d 1] = [X Y Z W]������dΪ��������ͼ���ʱ��

	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q,
		CALIB_ZERO_DISPARITY, -1, imageSize, &validROIL, &validROIR);

	//����stereoRectify ���������R �� P ������ͼ���ӳ��� mapx,mapy
	//mapx,mapy������ӳ������������Ը�remap()�������ã���У��ͼ��ʹ������ͼ���沢���ж�׼
	//ininUndistortRectifyMap()�Ĳ���newCameraMatrix����У����������������openCV���棬У����ļ��������Mrect�Ǹ�ͶӰ����Pһ�𷵻صġ�
	//���������ﴫ��ͶӰ����P���˺������Դ�ͶӰ����P�ж���У��������������

	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);


	Mat rectifyImageL, rectifyImageR;
	cvtColor(grayImageL, rectifyImageL, CV_GRAY2BGR);
	cvtColor(grayImageR, rectifyImageR, CV_GRAY2BGR);

	//imshow("Rectify Before", rectifyImageL);

	//remap�����������ͼ���沢���ж�׼
	remap(rectifyImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
	remap(rectifyImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

	//imshow("ImageL", rectifyImageL);
	//imshow("ImageR", rectifyImageR);

	//����ͼ�񻭵�������
	Mat canvasPart = canvas(Rect(0, h * 2, w, h));
	resize(rectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);

	//����ͼ�񻭵�������
	canvasPart = canvas(Rect(w, h * 2, w, h));
	resize(rectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
	putText(canvas, "LeftRectified", cv::Point(5, 495), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
	putText(canvas, "RightRectified", cv::Point(325, 495), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
	imshow("Output", canvas);

	//���������ͼ��
	//imwrite("ImageL",rectifyImageL);
	//imwrite("ImageR", rectifyImageR);


	/*���沢�������*/
	outputCameraParam();


	//��У�������ʾ����
	//����������ͼ����ʾ��ͬһ��������
	//����ֻ��ʾ�����һ��ͼ���У���������û�а����е�ͼ����ʾ����

	Mat canvas_;
	double sf_;
	int w_, h_;
	sf_ = 600. / MAX(imageSize.width, imageSize.height);
	w_ = cvRound(imageSize.width * sf_);
	h_ = cvRound(imageSize.height * sf_);
	canvas_.create(h_, w_ * 2, CV_8UC3);

	//��ͼ�񻭵�������
	Mat canvasPart_ = canvas_(Rect(w_ * 0, 0, w_, h_));                                //�õ�������һ����  
	resize(rectifyImageL, canvasPart_, canvasPart_.size(), 0, 0, INTER_AREA);  //��ͼ�����ŵ���canvasPartһ����С  
	Rect vroiL(cvRound(validROIL.x*sf_), cvRound(validROIL.y*sf_),                //��ñ���ȡ������    
		cvRound(validROIL.width*sf_), cvRound(validROIL.height*sf_));
	rectangle(canvasPart_, vroiL, Scalar(0, 0, 200), 3, 8);                      //����һ������  

	cout << "Painted ImageL" << endl;

	//��ͼ�񻭵�������
	canvasPart_ = canvas_(Rect(w_, 0, w_, h_));                                      //��û�������һ����  
	resize(rectifyImageR, canvasPart_, canvasPart_.size(), 0, 0, INTER_LINEAR);
	Rect vroiR(cvRound(validROIR.x * sf_), cvRound(validROIR.y*sf_),
		cvRound(validROIR.width * sf_), cvRound(validROIR.height * sf_));
	rectangle(canvasPart_, vroiR, Scalar(0, 159, 0), 3, 8);

	cout << "Painted ImageR" << endl;

	//���϶�Ӧ������
	for (int i = 0; i < canvas_.rows;i += 16)
		line(canvas_, Point(0, i), Point(canvas_.cols, i), Scalar(0, 200, 0), 1, 8);

	imshow("Rectified", canvas_);

	cout << "wait key" << endl;
	waitKey(0);
	system("pause");
	return 0;
}