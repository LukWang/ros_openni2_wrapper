///////////////////////////////////////////////////////////////////////////////
//
// Simple program that reads depth and color (RGB) images from Primensense
// camera using OpenNI2 and displays them using OpenCV.
//
// Ashwin Nanjappa
///////////////////////////////////////////////////////////////////////////////

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
//#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
//#include <opencv2/gpu/gpu.hpp>
//#include <opencv2/gpu/stream_accessor.hpp>

//#include <aruco/aruco.h>
//#include <aruco/cvdrawingutils.h>


#include <OpenNI.h>

#include <string>
#include <iostream>
#include <sstream>

#include <pthread.h>

#include <depth_registration.h>
using namespace cv;
using namespace std;
//using namespace aruco;

//extern "C" void register_depth_CUDA(const PtrStepSz<ushort1>& depth_mat, PtrStep<ushort1> registered_mat, const PtrStepSz<double>& color_camera_matrix, PtrStepSz<double>& rot, PtrStepSz<double>& trans, cudaStream stream);

class Grabber
{
public:
	Grabber(const ros::NodeHandle& );
    void InitOpenNI();
    void InitDevice();
		bool InitCalibration();
    void InitDepthStream();
    void InitColorStream();
    void Run();
		void Exit();
		void ShowFps();
		void Screenshot();
		void Combine();
    void NodeCombine();
    void PublishDepth();
		void calibrateDepthFrame(cv::Mat &, cv::Mat&);
		void loadCalibrationFiles();
private:
    void CapturePsenseDepthFrame();
    void CapturePsenseColorFrame();

	//void locateMarker(cv::Mat &color_mat);

    static void* processColorMat(void* color_mat);
	cv::Mat ChangeDepthForDisplay(const cv::Mat& mat);

	stringstream ss;

    openni::Device*        device_;
    openni::VideoStream*   depth_stream_;
    openni::VideoStream*   color_stream_;
    openni::VideoFrameRef* depth_frame_;
    openni::VideoFrameRef* color_frame_;

	DepthRegistrationOpenCL* depthReg;
	cv::Size sizeColor, sizeColorClipped, sizeDepth;
	cv::Mat cameraMatrix_color, cameraMatrix_color_clipped, cameraMatrix_depth, distortion_color, distortion_depth;
	cv::Mat cam_rot, cam_trans;
	cv::Mat map1_color, map2_color;
	cv::Mat color_mat;
	cv::Mat depth_mat;
	//cv::Mat combined_mat;
    ros::NodeHandle node_handle;
    image_transport::ImageTransport it;
    image_transport::Publisher pub ;
    image_transport::Publisher pub2;
	double depthShift;
	bool saveColorFrame;
	string frame_save_path;
	int color_frame_count;
    double timeFPS;

	//MarkerDetector marker_detector;
	//vector<Marker> markers;
};

Grabber::Grabber(const ros::NodeHandle &node_handle = ros::NodeHandle()) : sizeColor(1920, 1080), sizeColorClipped(sizeColor.width / 2, sizeColor.height / 2), sizeDepth(512, 424),it(node_handle)
{
    depth_mat = cv::Mat(sizeColor, CV_16U);
    pub = it.advertise("camera/image", 1);
    pub2 = it.advertise("camera/depth_image",1);
    saveColorFrame = false;
	timeFPS = (double)getTickCount();
	frame_save_path = "/home/luk/catkin_ws/src/camera_wrapper/frames";
	//cv::Directory frame_dir;
	//std::vector<string> frame_names = frame_dir.GetListFiles(frame_save_path, "*.jpg", false);
	color_frame_count = 0;
	cout << "Exisiting color frame: " + color_frame_count;

}

struct thread_color_params{
    Grabber* pThis;
    Mat* color_mat_ptr;
};

void Grabber::InitOpenNI()
{
    auto rc = openni::OpenNI::initialize();
    if (rc != openni::STATUS_OK)
    {
        printf("Initialize failed\n%s\n", openni::OpenNI::getExtendedError());
        exit(0);
    }
	//openni::VideoStream::setMirroringEnabled(false);
}

void Grabber::InitDevice()
{
    device_ = new openni::Device();
    auto rc = device_->open(openni::ANY_DEVICE);
    if (rc != openni::STATUS_OK)
    {
        printf("Couldn't open device\n%s\n", openni::OpenNI::getExtendedError());
        exit(0);
    }
}

bool Grabber::InitCalibration()
{
	 const int mapType = CV_16SC2;
	 loadCalibrationFiles();
	depthReg = new DepthRegistrationOpenCL();

	cv::initUndistortRectifyMap(cameraMatrix_color, distortion_color, cv::Mat(), cameraMatrix_color, sizeColor, mapType, map1_color, map2_color);


	if(!depthReg->init(cameraMatrix_color_clipped, sizeColorClipped, cameraMatrix_depth, sizeDepth, distortion_depth, cam_rot, cam_trans, 0.5f, 12.0f,1))
		return true;
	else
	{
		printf("Calibration initiation failed!");
		return false;
	}
}

void Grabber::InitDepthStream()
{
    depth_stream_ = new openni::VideoStream();
	//depth_stream_->setMirroringEnabled(true);
    // Create depth stream from device
    if (device_->getSensorInfo(openni::SENSOR_DEPTH) != nullptr)
    {
        auto rc = depth_stream_->create(*device_, openni::SENSOR_DEPTH);
        if (rc != openni::STATUS_OK)
        {
            printf("Couldn't create depth stream\n%s\n", openni::OpenNI::getExtendedError());
            exit(0);
        }
    }
	cout << "max depth value " <<depth_stream_->getMaxPixelValue() << endl;
	cout << "max depth value " <<depth_stream_->getMinPixelValue() << endl;
    // Get info about depth sensor
    const openni::SensorInfo& sensor_info       = *device_->getSensorInfo(openni::SENSOR_DEPTH);
    const openni::Array<openni::VideoMode>& arr = sensor_info.getSupportedVideoModes();

    // Look for VGA mode in depth sensor and set it for depth stream
    for (int i = 0; i < arr.getSize(); ++i)
    {
        const openni::VideoMode& vmode = arr[i];
        printf("%i: %ix%i, %i fps, %i format\n", i, vmode.getResolutionX(), vmode.getResolutionY(),vmode.getFps(), vmode.getPixelFormat());
        if (vmode.getPixelFormat() == openni::PIXEL_FORMAT_DEPTH_1_MM &&
            vmode.getResolutionX() == 512 &&
            vmode.getResolutionY() == 424)
        {
            depth_stream_->setVideoMode(vmode);
            break;
        }
    }

    // Start the depth stream
    auto rc = depth_stream_->start();
    if (rc != openni::STATUS_OK)
    {
        printf("Couldn't start the depth stream\n%s\n", openni::OpenNI::getExtendedError());
        exit(0);
    }

    depth_frame_ = new openni::VideoFrameRef();
}

void Grabber::InitColorStream()
{
    color_stream_ = new openni::VideoStream();
	//color_stream_->setMirroringEnabled(false);

    if (device_->getSensorInfo(openni::SENSOR_COLOR) != nullptr)
    {
        auto rc = color_stream_->create(*device_, openni::SENSOR_COLOR);
        if (rc != openni::STATUS_OK)
        {
            printf("Couldn't create color stream\n%s\n", openni::OpenNI::getExtendedError());
            exit(0);
        }
    }

    // Get info about color sensor
    const openni::SensorInfo& sensor_info = *device_->getSensorInfo(openni::SENSOR_COLOR);
    const openni::Array<openni::VideoMode>& arr = sensor_info.getSupportedVideoModes();

    // Look for VGA mode and set it for color stream
    for (int i = 0; i < arr.getSize(); ++i)
    {
		const openni::VideoMode& vmode = arr[i];
        printf("%i: %ix%i, %i fps, %i format\n", i, vmode.getResolutionX(), vmode.getResolutionY(),vmode.getFps(), vmode.getPixelFormat());
        if (
            vmode.getResolutionX() == 1920 &&
            vmode.getResolutionY() == 1080)
        {
            color_stream_->setVideoMode(vmode);
            break;
        }
    }

    // Note: Doing image registration earlier than this seems to fail
 /**
   	if (device_->isImageRegistrationModeSupported(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR))
    {
        auto rc = device_->setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
        if (rc == openni::STATUS_OK)
            std::cout << "Depth to color image registration set success\n";
        else
            std::cout << "Depth to color image registration set failed\n";
    }
    else
    {
        std::cout << "Depth to color image registration is not supported!!!\n";
    }
    **/

    // Start color stream
    auto rc = color_stream_->start();
    if (rc != openni::STATUS_OK)
    {
        printf("Couldn't start the depth stream\n%s\n", openni::OpenNI::getExtendedError());
        exit(0);
    }

    color_frame_ = new openni::VideoFrameRef();
}
void Grabber::ShowFps()
{
	 //Display FPS
	timeFPS = ((double)cv::getTickCount() - timeFPS) / cv::getTickFrequency();
    short fps = (short)(1.0 / timeFPS);
    stringstream tempss;
    tempss << fps;
    cout << "[fps:" << fps << "]" << endl;
    cout << "[time:" << timeFPS << "]" << endl;
    putText(color_mat, tempss.str(), cv::Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
}
void Grabber::Screenshot()
{
		color_frame_count++;
		ss << color_frame_count;

		cv::imwrite(frame_save_path + "color_frame_" + ss.str() + ".jpg", color_mat);

}
void Grabber::Combine()
{		 //cv::flip(color_mat, color_mat, 1);
	// cv::flip(depth_mat, depth_mat, 1);
		    //Combining RGB frame with depth frame
		//cv::Mat combined_mat;
		cv::Mat combined_mat = color_mat;

		for (int r = 0;r < depth_mat.rows; ++r)
		    for (int c = 0;c < depth_mat.cols; ++c)
		{
			if (depth_mat.at<uint16_t>(r, c) > 3000 || depth_mat.at<uint16_t>(r, c) == 0)
				combined_mat.at<cv::Vec3b>(r, c) = cv::Vec3b(0,255,0);
		}
		//cv::imshow("color_mat",combined_mat);

}
void Grabber::NodeCombine()
{
    sensor_msgs::ImagePtr msg;
    if(pub.getNumSubscribers() > 0)
    {

        msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color_mat).toImageMsg();
        pub.publish(msg);
     } //  cv::waitKey(1);

    //ros::spinOnce();
    //loop_rate.sleep();
}
void Grabber::PublishDepth()
{
    sensor_msgs::ImagePtr msgdepth;
    if(pub2.getNumSubscribers() > 0)
    {

        msgdepth = cv_bridge::CvImage(std_msgs::Header(), "mono16", depth_mat).toImageMsg();
        pub2.publish(msgdepth);
     } //  cv::waitKey(1);
}
void Grabber::CapturePsenseDepthFrame()
{
    auto rc = depth_stream_->readFrame(depth_frame_);
    if (rc != openni::STATUS_OK)
    {
        printf("Read failed!\n%s\n", openni::OpenNI::getExtendedError());
    }

    if (depth_frame_->getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_DEPTH_1_MM && depth_frame_->getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_DEPTH_100_UM)
    {
        printf("Unexpected frame format\n");
    }

    // Get pointer to Primesense depth frame
    openni::DepthPixel* dev_buf_ptr = (openni::DepthPixel*) depth_frame_->getData();

    // Copy frame data to OpenCV mat
  cv::Mat depth_mat_tmp=cv::Mat(depth_frame_->getHeight(), depth_frame_->getWidth(), CV_16U, dev_buf_ptr);
	//cv::remap(color_mat, color_mat, map1_color, map2_color, cv::INTER_AREA);
    //depth_mat_tmp.convertTo(depth_mat_tmp, CV_16U);
	cv::flip(depth_mat_tmp, depth_mat_tmp, 1);
	depth_mat_tmp.convertTo(depth_mat_tmp, CV_16U, 1, depthShift);
	calibrateDepthFrame(depth_mat_tmp, depth_mat);
	//cv::Mat disp_mat = ChangeDepthForDisplay(depth_mat);
	//disp_mat = disp_mat(cv::Range(32,392), cv::Range(0, 512));

  //cv::imshow("Depth", disp_mat);
}

void Grabber::CapturePsenseColorFrame()
{
    // Read from stream to frame
    auto rc = color_stream_->readFrame(color_frame_);
    if (rc != openni::STATUS_OK)
    {
        printf("Read failed!\n%s\n", openni::OpenNI::getExtendedError());
    }

    // Pointer to Primesense color frame
   // openni::RGB888Pixel* dev_buf_ptr = (openni::RGB888Pixel*) color_frame_->getData();

    // Make mat from camera data
     color_mat = cv::Mat(color_frame_->getHeight(), color_frame_->getWidth(), CV_8UC3, (void*)color_frame_->getData());
     cv::resize(color_mat, color_mat, sizeColorClipped);
     cv::flip(color_mat, color_mat, 1);
		 cv::cvtColor(color_mat, color_mat, CV_RGB2BGR);
    // Convert to BGR format for OpenCV
    //processColorMat((void*) &color_mat);
    //Multithread process of RGB frame
   /***
    pthread_t thread_colorProcess;
    thread_color_params thread_param;
    thread_param.pThis = this;
    thread_param.color_mat_ptr = &color_mat;
    int ret = pthread_create(&thread_colorProcess, NULL, processColorMat, (void*) &thread_param);
***/
    //reading depth frame

    //Wait for RGB frame processing thread join in
    //pthread_join(thread_colorProcess, NULL);

    //Save the RGB frame if 's' is pressed




    //cv::imshow("Color", color_mat);
}

void Grabber::calibrateDepthFrame(cv::Mat &depth_mat_tmp, cv::Mat &depth_mat)
{
	depthReg->registerDepth(depth_mat_tmp, depth_mat);
    printf("image registered");
}
void* Grabber::processColorMat(void* params)
{
    //*((Mat*)color_mat)
    thread_color_params *params_ptr = (thread_color_params*)params;
    cv::cvtColor(*(params_ptr->color_mat_ptr), *(params_ptr->color_mat_ptr), CV_RGB2BGR);
	cv::flip(*(params_ptr->color_mat_ptr), *(params_ptr->color_mat_ptr), 1);
	cv::resize(*(params_ptr->color_mat_ptr), *(params_ptr->color_mat_ptr), cv::Size((params_ptr->pThis)->sizeColorClipped.width, (params_ptr->pThis)->sizeColorClipped.height));
	//(params_ptr->pThis)->locateMarker( *(params_ptr->color_mat_ptr));
    return NULL;
}
/**
void Grabber::locateMarker(cv::Mat &color_mat)
{
	marker_detector.detect(color_mat, markers);
	//cout << markers.size() << endl;
	for (int i = 0; i < markers.size(); i++)
		markers[i].draw(color_mat, cv::Scalar(0, 0, 255), 2);
}
**/
void Grabber::loadCalibrationFiles()
{
	cv::FileStorage fs;
	string calib_path = "/home/agent/catkin_ws/src/camera_wrapper/calibration_data/003415165047/";

	if(fs.open(calib_path + "calib_color.yaml", cv::FileStorage::READ))
	{
		fs["cameraMatrix"] >> cameraMatrix_color;
		cameraMatrix_color_clipped = cameraMatrix_color.clone();
		cameraMatrix_color_clipped.at<double>(0, 0) /= 2;
		cameraMatrix_color_clipped.at<double>(1, 1) /= 2;
		cameraMatrix_color_clipped.at<double>(0, 2) /= 2;
		cameraMatrix_color_clipped.at<double>(1, 2) /= 2;
		fs["distortionCoefficients"] >> distortion_color;
		cout << "color matrix load success"<< endl;
		fs.release();
	}
	else
	{
		cout << "No calibration file: calib_color.yalm, using default calibration setting" << endl;
		cameraMatrix_color = cv::Mat::eye(3, 3, CV_64F);
		distortion_color = cv::Mat::zeros(1, 5, CV_64F);
	}

	if(fs.open(calib_path + "calib_ir.yaml", cv::FileStorage::READ))
	{
		fs["cameraMatrix"] >> cameraMatrix_depth;
		fs["distortionCoefficients"] >> distortion_depth;
		cout << "ir matrix load success"<< endl;
		fs.release();
	}
	else
	{
		cout << "No calibration file: calib_ir.yalm, using default calibration setting" << endl;
		cameraMatrix_depth = cv::Mat::eye(3, 3, CV_64F);
		distortion_depth = cv::Mat::zeros(1, 5, CV_64F);
	}

	if(fs.open(calib_path + "calib_pose.yaml", cv::FileStorage::READ))
	{
		fs["rotation"] >> cam_rot;
		fs["translation"] >> cam_trans;
		cout << "camera pose load success"<< endl;
		fs.release();
	}
	else
	{
		cout << "No calibration file: calib_color.yalm, using default calibration setting" << endl;
		cam_rot = cv::Mat::eye(3, 3, CV_64F);
		cam_trans = cv::Mat::zeros(3, 1,CV_64F);
	}

	if(fs.open(calib_path + "calib_depth.yaml", cv::FileStorage::READ))
	{
		fs["depthShift"] >> depthShift;
		cout << "color matrix load success"<< endl;
		fs.release();
	}
	else
	{
		cout << "No calibration file: calib_color.yalm, using default calibration setting" << endl;
		depthShift = 0.0;
	}
}



void Grabber::Run()
{
    openni::VideoStream* streams[] = {depth_stream_, color_stream_};
    bool doCombine = false;
    ros::Rate rate(100);
   while (ros::ok())
   {
       int readyStream = -1;
	   timeFPS = (double)getTickCount();
       auto rc = openni::OpenNI::waitForAnyStream(streams, 2, &readyStream, 2000);
       if (rc != openni::STATUS_OK)
       {
           printf("Wait failed! (timeout is %d ms)\n%s\n", 2000, openni::OpenNI::getExtendedError());
           break;
       }

       switch (readyStream)
       {
       case 0:
        CapturePsenseDepthFrame();
        CapturePsenseColorFrame();
           break;
       case 1:
        CapturePsenseColorFrame();
        CapturePsenseDepthFrame();

           break;
       default:
           printf("Unxpected stream\n");
       }
        if(!color_mat.empty() && !depth_mat.empty())
        {
          NodeCombine();
          PublishDepth();
        	ShowFps();
        	if(doCombine)
        	{
        		Combine();
        	}
            //else
                //imshow("color_mat", color_mat);
         }
        char c = cv::waitKey(10);
        if ('q' == c)
		{
            break;
		}
		else if ('s' == c)
			Screenshot();
		else if ('c' == c)
		{
			doCombine = !doCombine;

		}
        rate.sleep();
   }
}

cv::Mat Grabber::ChangeDepthForDisplay(const cv::Mat& mat)
{
    assert(CV_16U == mat.type());

    const float depth_near = 500;
    const float depth_far  = 5000;

    const float alpha = 255.0 / (depth_far - depth_near);
    const float beta  = - depth_near * alpha;

    cv::Mat fmat;
    mat.convertTo(fmat, CV_32F);

    for (int r = 0; r < mat.rows; ++r)
    {
        for (int c = 0; c < mat.cols; ++c)
        {
            float v = fmat.at<float>(r, c) * alpha + beta;

            if (v > 255) v = 255;
            if (v < 0)   v = 0;

            fmat.at<float>(r, c) = v;
        }
    }

    cv::Mat bmat;
    fmat.convertTo(bmat, CV_8U);

    cv::Mat cmat;
    cv::cvtColor(bmat, cmat, CV_GRAY2BGR);
    cv::applyColorMap(cmat, cmat, cv::COLORMAP_OCEAN);

    return cmat;
}

void Grabber::Exit()
{
	delete depthReg;
	delete depth_stream_;
	delete color_stream_;
	device_->close();
	delete device_;
	openni::OpenNI::shutdown();
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_publisher");
    Grabber grabber;
    grabber.InitOpenNI();
    grabber.InitDevice();
	  grabber.InitCalibration();
    grabber.InitDepthStream();
    grabber.InitColorStream();

    grabber.Run();
    grabber.Exit();
    return 0;
}
