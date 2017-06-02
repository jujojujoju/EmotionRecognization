#include <iostream>
#include <string>
#include <dirent.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

//#include "flandmark_detector.h"
using namespace std;
using namespace cv;

vector<string> listFile(string folder) {
    vector<string> imgPath;
    DIR *pDIR;
    struct dirent *entry;
    if (pDIR = opendir(folder.c_str())) {
        while (entry = readdir(pDIR)) {
            if (entry->d_type == DT_REG) {        // if entry is a regular file
                std::string fname = entry->d_name;    // filename
//                std::string::size_type size = fname.find(".tiff");
                std::string::size_type size = fname.find(".png");

                if (size != std::string::npos) {
                    imgPath.push_back(folder + "/" + fname);
                }
            }
        }
    }
    return imgPath;
}

int main() {

    string input = "/Users/joju/Downloads/image_data";
    vector<string> imgPath = listFile(input);

    int num_of_image = (int) imgPath.size();

    int tailNum = 0;
    int subTailNum = 0;

    int middleNum = 0;
    int subMiddleNum = 1;

    int firstNum = 0;
    int subFirstNum = 1;
    for (int img_id = 0; img_id < num_of_image; img_id++) {

        string fileName = imgPath[img_id].substr(input.length() + 1, imgPath[img_id].length());

        cout << fileName << endl;

        middleNum = atoi(fileName.substr(5, 3).c_str());

        tailNum = atoi(fileName.substr(14, 3).c_str());

        firstNum = atoi(fileName.substr(1,3).c_str());
        cout<<firstNum<<endl;


        if (subMiddleNum != middleNum || firstNum != subFirstNum) {
            cout << endl << "표정 혹은 사람 변경 !!!!!!!" << endl << endl;
            if(img_id>6) {
                for (int i = 0; i < 4; i++) {
                    string currentImagePath = imgPath[img_id-1 - i*2];
                    string prefilename = currentImagePath.substr(input.length() + 1, currentImagePath.length());
                    cout << "prefilename : " << prefilename << endl;
                    imwrite( "/Users/joju/OpenCVBlueprints/chapter_3/filterdImage/" + prefilename,
                             imread(currentImagePath, CV_LOAD_IMAGE_COLOR));
                }
            }
            cout<<endl;
        }

        subFirstNum = firstNum;
        subMiddleNum = middleNum;

        cout << "middleNum : " << middleNum << endl;
        cout << "tailNum : " << tailNum << endl;


        if (img_id == 10700)
            break;
    }


    return 0;
}

