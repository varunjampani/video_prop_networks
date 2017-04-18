
#include <time.h>
#include <stdio.h>
#include <fstream>

#include <boost/filesystem.hpp>

#include "gSLICr_Lib/gSLICr.h"
#include "NVTimer.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void load_image(const Mat& inimg, gSLICr::UChar4Image* outimg)
{
	gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < outimg->noDims.y;y++)
		for (int x = 0; x < outimg->noDims.x; x++)
		{
			int idx = x + y * outimg->noDims.x;
			outimg_ptr[idx].b = inimg.at<Vec3b>(y, x)[0];
			outimg_ptr[idx].g = inimg.at<Vec3b>(y, x)[1];
			outimg_ptr[idx].r = inimg.at<Vec3b>(y, x)[2];
		}
}

void load_image(const gSLICr::UChar4Image* inimg, Mat& outimg)
{
	const gSLICr::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < inimg->noDims.y; y++)
		for (int x = 0; x < inimg->noDims.x; x++)
		{
			int idx = x + y * inimg->noDims.x;
			outimg.at<Vec3b>(y, x)[0] = inimg_ptr[idx].b;
			outimg.at<Vec3b>(y, x)[1] = inimg_ptr[idx].g;
			outimg.at<Vec3b>(y, x)[2] = inimg_ptr[idx].r;
		}
}

int main(int argc, char* argv[])
{

        if (argc < 5)
            std::cout << "Usage " << argv[0] << " image_dir image_list out_dir num_spixels\n";

        boost::filesystem::create_directories(argv[3]);

	StopWatchInterface *my_timer; sdkCreateTimer(&my_timer);

	// gSLICr settings
	gSLICr::objects::settings my_settings;
	my_settings.no_segs = atoi(argv[4]); //1000 superpixels
	my_settings.spixel_size = 16;
	my_settings.coh_weight =  1.5f; // 0.6f;
	my_settings.no_iters = 5;
	my_settings.color_space = gSLICr::CIELAB; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB XYZ for
	my_settings.seg_method = gSLICr::GIVEN_NUM; // or gSLICr::GIVEN_SIZE for given number
	my_settings.do_enforce_connectivity = true; // wheter or not run the enforce connectivity step

	int save_count = 0;
        string line;
	ifstream infile(argv[2]);
	while(infile>> line){
		string imgname = string(argv[1]) + line + ".jpg";
		Mat image = imread(imgname,1);

		// gSLICr settings
		my_settings.img_size.x = image.cols;
		my_settings.img_size.y = image.rows;
		gSLICr::engines::core_engine* gSLICr_engine = new gSLICr::engines::core_engine(my_settings);
		// gSLICr takes gSLICr::UChar4Image as input and out put
		gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
		gSLICr::UChar4Image* out_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
		Size s(my_settings.img_size.x, my_settings.img_size.y);
		Mat boundry_draw_frame; boundry_draw_frame.create(s, CV_8UC3);

                load_image(image, in_img);
		sdkResetTimer(&my_timer);
                sdkStartTimer(&my_timer);
		gSLICr_engine->Process_Frame(in_img);
		sdkStopTimer(&my_timer);
		cout << save_count << ". " << line.c_str() << " in "
                     << sdkGetTimerValue(&my_timer)<<"ms.\n";

		gSLICr_engine->Draw_Segmentation_Result(out_img);
		load_image(out_img, boundry_draw_frame);

		char out_name[100];
		sprintf(out_name, "%s%s.pgm", argv[3], line.c_str());
		boost::filesystem::path out_f(out_name);
		boost::filesystem::path out_dir = out_f.parent_path();
		if(!(boost::filesystem::exists(out_dir))){
        boost::filesystem::create_directory(out_dir);
    }
		gSLICr_engine->Write_Seg_Res_To_PGM(out_name);
		save_count++;

		delete in_img;
		delete out_img;
		delete gSLICr_engine;
	}

	return 0;
}
