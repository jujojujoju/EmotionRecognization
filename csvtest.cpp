#include <iostream>
#include <string>
#include <dirent.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "flandmark_detector.h"

using namespace std;
using namespace cv;

void processJAFFE(string input, string output);
vector<string> listFile(string folder);
CascadeClassifier loadCascade(string cascadePath);

int main(int argc, char* argv[]) {
    cout << "============= FACIAL COMPONENTS =============" << endl;
    if(argc != 5){
        cout << "Usage: " << endl;
        cout << argv[0] << " -num <num_of_image> -dest <output_folder>" << endl;
        return 1;
    }

    // ********************
    // Get input parameters
    // ********************

    string input;
    string output_folder;

    for(int i = 0; i < argc ; i++)
    {
        if( strcmp(argv[i], "-num") == 0 ){
            if(i + 1 >= argc) return -1;
            input = argv[i + 1];
        }

        if( strcmp(argv[i], "-dest") == 0 ){
            if(i + 1 >= argc) return -1;
            output_folder = argv[i + 1];
        }
    }

    // ********************
    // JAFFE Dataset
    // ********************
    processJAFFE(input, output_folder);

    return 0;
}

void processJAFFE(string input, string output) {
    cout << "Process JAFFE: " << input << endl;

//    int num_of_image = 1000;
    int num_of_image = atoi(input.c_str());

    ifstream file("/Users/joju/Downloads/fer2013/fer2013.csv");
    int img_id = 0;
    string current_line;

    ////
    FileStorage fs(output + "/list.yml", FileStorage::WRITE);
    fs << "num_of_image" << num_of_image;

    while (getline(file, current_line))
    {
        string emotion;
        vector<int> pixel_vector;

        if(img_id == num_of_image)
            break;

        stringstream currentLineStream(current_line);
        string single_Cell;

        int j = 0;
        while (getline(currentLineStream, single_Cell, ','))
        {
            if(j == 0)
            {
//                emotion.push_back(single_Cell);
                emotion = single_Cell;
//                if(single_Cell == "0"){
//                    emotion = 0;
//                } else if(single_Cell == "1"){
//                    emotion = 1;
//                } else if(single_Cell == "2"){
//                    emotion = 2;
//                } else if(single_Cell == "3"){
//                    emotion = 3;
//                } else if(single_Cell == "6"){
//                    emotion = 4;
//                } else if(single_Cell == "4"){
//                    emotion = 5;
//                } else if(single_Cell == "5"){
//                    emotion = 6;
//                }
            }
            else if (j == 1)
            {
                stringstream currentCellStream(single_Cell);
                string pixel_value;
                while (getline(currentCellStream, pixel_value, ' '))
                {
                    pixel_vector.push_back( atoi(pixel_value.c_str()));
//                    pixel_vector.push_back(pixel_value);
                }
            }
            j++;
        }

        Mat test1 = Mat::zeros(48, 48, CV_8UC1);
        for (int rows = 0; rows < 48; rows++) {
            for (int cols = 0; cols < 48; cols++) {
                test1.at<uchar>(rows, cols) = (uchar) pixel_vector[rows * 48 + cols];
            }
        }

//        resize(test1,test1,Size(256,256));
        cvtColor(test1,test1,CV_GRAY2RGB);


        resize(test1,test1,Size(160,160));

        test1.convertTo(test1,CV_8UC3);
        cout<<test1.type()<<endl;
        cout<<test1.elemSize1()<<endl;

        string fileName = "EM"+emotion+'_'+to_string(img_id)+"_face.png";
        imwrite(output + "/" + fileName, test1);
        ////
        fs << "img_" + to_string(img_id) + "_face" << output + "/" + fileName;

        img_id++;
    }
    ////
    fs << "real_num_of_image" << num_of_image;
}

CascadeClassifier loadCascade(string cascadePath){
    CascadeClassifier cascade;
    if (!cascade.load(cascadePath)) {
        cout << "--(!)Error loading cascade: " << cascadePath << endl;
    };
    return cascade;
}
