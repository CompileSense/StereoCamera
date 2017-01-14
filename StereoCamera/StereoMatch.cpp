//双目匹配
//版本:Version 3.3.1
//利用双目标定文件对双目摄像头所采得的画面进行匹配并能够输出图像世界坐标
//最新更新：在OPENGL窗口中实时分类显示三维特征点
//备注：需自行添加模型库和训练结果

#include <iostream>

//OPENCV相关头文件
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <vector>
#include <stdio.h>

//面部角度检测器
#include "facedetect-dll.h"
#pragma comment(lib,"libfacedetect-x64.lib")

//脸部识别头文件
#include "LBF_api.h"
using namespace cv;
using namespace std;

//OPENGL相关头文件
#include <GL/glut.h>

//存放特征点坐标
vector<Point3f> points;
float xyzdata[68][3];

Rect roi1, roi2;
Mat Q;
float scale = 0.5;
Mat canvas;
int w = 320, h = 240;
Mat M1, D1, M2, D2;
Mat R, T, R1, P1, R2, P2;
Mat disp, disp8, img1, img2;
CascadeClassifier cascade, nestedCascade;
int alg = 1;

//匹配参数
int SADWindowSize = 1, numberOfDisparities = 128;
////////////////////5////////////////////////256/
bool no_display = false;

//鼠标响应初始化
GLboolean mousedown = GL_FALSE;
//鼠标位置初始化
static GLint mousex = 0, mousey = 0;
//中心点位置初始化
static GLfloat center[3] = { 0.0f, 0.0f, 0.0f };
//观察位置初始化
static GLfloat eye[3] = { 0.0f, 0.0f, -20.0f };
//观察角度初始化
static GLfloat yrotate = 0;
static GLfloat xrotate = 0;

Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);

int * pResults = NULL;
Mat left, right, xyz, gray;
float angle = 0;
//打开左摄像头
VideoCapture capleft(1);
//打开右摄像头
VideoCapture capright(0);



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

void detectAndDraw(Mat& inputimg,Mat& dirimg,Mat& depthimg,CascadeClassifier& cascade,
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
		
		int error = 0;
		//FILE* fp = fopen("68points.txt", "wt");
		int number = 0;
		while (p_point != landmarks.end())
		{
			center.x = cvRound((r->x + p_point->x)*scale);
			center.y = cvRound((r->y + p_point->y)*scale);
			//circle(dirimg, center, 3, color, 1, 8, 0);
			p_point++;
			char t[10];
			string s;
			sprintf(t, "%d", number);
			s = t;
			putText(dirimg, s, Point(center.x,center.y), CV_FONT_HERSHEY_COMPLEX, 0.25, CV_RGB(255, 255, 255), 1, 8);

			Vec3f point,cpoint;
			int wrong = 0;
			for(int p=0;p<5;p++)
				for (int q = 0;q < 5;q++)
				{
					cpoint = depthimg.at<Vec3f>(center.y + q-2, center.x + p-2);
					if (fabs(cpoint[2] - 80) < FLT_EPSILON || fabs(cpoint[2]) > 80)
					{
						wrong++;
					}
					else
					{
						point = cpoint + point;
					}
				}
			if (wrong == 25)
			{
				//error++;
				xyzdata[number][0] = 0;
				xyzdata[number][1] = 0;
				xyzdata[number][2] = 0;
				number++;
				continue;
			}
			point = point / (25 - wrong);
			//将坐标写入点云文件
			//fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
			//将特征点存入points矩阵
			//points.push_back(Point3f(point[0], point[1], point[2]));
			xyzdata[number][0] = point[0];
			xyzdata[number][1] = point[1];
			xyzdata[number][2] = point[2]-40;
			number++;
		}
	
		//cout << "Wrong point: " << error << endl;
		
	}
	imshow("Result",dirimg);
	//return points;
}

void display(void)
{
	Mat left, right;
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

	//缩小一倍增加检测速度
	resize(left, img1, left.size() / 2, 0, 0, INTER_AREA);
	resize(right, img2, right.size() / 2, 0, 0, INTER_LINEAR);
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
	sgbm->setUniquenessRatio(15);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setSpeckleRange(32);
	sgbm->setDisp12MaxDiff(1);
	sgbm->setMode(StereoSGBM::MODE_SGBM);


	//显示运行时间
	int64 t = getTickCount();
	//SGBM计算
	sgbm->compute(img1, img2, disp);
	t = getTickCount() - t;
	cout << "Time elapsed: " << t * 1000 / getTickFrequency() << "ms" << endl;

	//将深度图解码为可视化图像
	disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));

	imshow("Depthwindow", disp8);

	//生成三维数组
	reprojectImageTo3D(disp, xyz, Q, true);

	//人脸角度探测函数
	cvtColor(img1, gray, CV_BGR2GRAY);
	pResults = facedetect_multiview_reinforce((unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, gray.step,
		1.2f, 5, 24);
	//printf("%d faces detected.\n", (pResults ? *pResults : 0));
	short * p = ((short*)(pResults + 1));
	if (p[5] == 0)angle = angle;
	else angle = p[5];

	cout << "Angle = " << angle << endl;
	//输出脸部角度探测结果
	//for (int i = 0; i < (pResults ? *pResults : 0); i++)
	//{
	//	short * p = ((short*)(pResults + 1)) + 6 * i;
	//	int x = p[0];
	//	int y = p[1];
	//	int w = p[2];
	//	int h = p[3];
	//	int neighbors = p[4];
	//	int angle = p[5];

	//	printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
	//}

	detectAndDraw(img1, disp8, xyz, cascade, nestedCascade, 1, 0);

	imshow("face", img1);

	if (cvWaitKey(10) == 'w')
	{
		printf("storing the point cloud...");
		saveXYZ("FacePointcloud.txt", xyz);
		printf("\n");
	};


	////OPENGL显示////

	//启用深度检测，否则后面的点云会遮盖前面的点云
	glEnable(GL_DEPTH_TEST);
	//清除颜色与深度缓存
	glClearColor(0.15, 0.15, 0.2, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//把当前矩阵设置为单位矩阵
	glLoadIdentity();
	//摄像机观察函数
	gluLookAt
	(
		eye[0], eye[1], eye[2],              //位置
		center[0], center[1], center[2],     //观察中心
		0, 1, 0                              //上向量
	);

	//将矩阵存入堆栈
	//glPushMatrix();

	//鼠标响应函数
	glRotatef(xrotate, 0.0, 1.0, 0.0);
	glRotatef(yrotate, 1.0, 0.0, 0.0);


	//调整点云位置及方向	
	glRotatef(180.0, 0.0, 0.0, 1.0);
	//glTranslatef(0, -5, -5);
	//glScaled(1.2f,1.2f,1.2f);
	float x, y, z;
	//绘制图像点云
	glPointSize(5.0f);
	glBegin(GL_POINTS);
	for (int i = 0;i<68;i++) {
		if (i - 17 < 0) glColor3f(1.0,0.0,0.0);
		else if (i - 36 < 0)glColor3f(0.0, 1.0, 0.0);
		else if (i - 48 < 0)glColor3f(0.0, 0.0, 1.0);
		else glColor3f(0.0, 1.0, 1.0);
			//读取点云坐标
			x = xyzdata[i][0];
			y = xyzdata[i][1];
			z = xyzdata[i][2];
			//多余点云过滤
			//if (z > 70 | z<5) continue;

			//glBegin(GL_TRIANGLE_STRIP);
			glVertex3f(x, y, z);
			//glEnd();
		
	}
	glEnd();

	//准线
	/*glLineWidth(5.0);
	glColor3f(1.0,1.0,1.0);
	glBegin(GL_LINES);
	glVertex3f(0, 0, -100);
	glVertex3f(0, 0, 100);
	glVertex3f(100, 0, 0);
	glVertex3f(-100, 0, 0);
	glVertex3f(0, 100, 0);
	glVertex3f(0, -100, 0);
	glEnd();*/

	//取出堆栈中的矩阵
	//glPopMatrix();

	//读取OPENGL像素信息并保存为Mat类
	//saveSceneImage();

	//交换双缓冲缓存
	glutSwapBuffers();
}


//鼠标响应函数
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mousedown = GL_TRUE;
	}
	mousex = x, mousey = y;
}

//鼠标控制视角函数
void motion(int x, int y)
{
	if (mousedown == GL_TRUE)
	{
		xrotate -= (x - mousex) / 5.0f;
		yrotate -= (y - mousey) / 5.0f;
	}
	mousex = x, mousey = y;
	glutPostRedisplay();
}

//键盘响应函数
void keyboard(unsigned char c, int x, int y)
{
	switch (c)
	{
	case 'w':
		eye[2] += 10.0f;

		break;
	case 's':
		eye[2] -= 10.0f;

		break;
	case 'a':
		eye[0] += 10.0f;
		break;
	case 'd':
		eye[0] -= 10.0f;
		break;
	case 'r':
		eye[0] = 0.0f;
		eye[2] = -45.0f;
		xrotate = 0;
		yrotate = 0;
		break;
	case 27:
		exit(0);
	default:
		break;
	}
	glutPostRedisplay();
}

//动画控制函数
void idle()
{
	glutPostRedisplay();
}

//摄像机注册函数
void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45, (GLfloat)w / (GLfloat)h, 1.0, 500.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

int main(int argc, char** argv)
{
	//读取yml双目匹配文件
	const char* intrinsic_filename = "data\\intrinsics.yml";
	const char* extrinsic_filename = "data\\extrinsics.yml";

	
	//读取面部特征点模型训练文件
	Initial_Model("D:\\Work\\Projects\\multcamera\\FaceAlignment\\FaceAlignment\\model\\");
	cascade.load("D:\\Work\\Projects\\multcamera\\FaceAlignment\\FaceAlignment\\test\\opencv-detection\\haarcascade_frontalface_alt.xml");
	nestedCascade.load("haarcascade_eye_tree_eyeglasses.xml");
	
	// 从data文件中读取标定参数
	FileStorage fs(intrinsic_filename, FileStorage::READ);

	fs["cameraMatrixL"] >> M1;
	fs["cameraDistcoeffL"] >> D1;
	fs["cameraMatrixR"] >> M2;
	fs["cameraDistcoeffR"] >> D2;
	M1 *= scale;
	M2 *= scale;

	fs.open(extrinsic_filename, FileStorage::READ);

	fs["R"] >> R;
	fs["T"] >> T;

	//绘制窗口

	canvas.create(h, w * 2, CV_8UC3);


	//OPENGL初始化函数
	glutInit(&argc, argv);

	//创建OPENGL窗口
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(100, 300);
	glutCreateWindow("3D_face_landmarkk_test");

	//注册OPENGL各个控制函数
	glutReshapeFunc(reshape);
	glutDisplayFunc(display);
	glutIdleFunc(idle);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	//运行主检测程序
	glutMainLoop();


		
	return 0;
}