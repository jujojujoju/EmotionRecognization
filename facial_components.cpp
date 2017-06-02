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

int main(int argc, char *argv[]) {
    cout << "============= FACIAL COMPONENTS =============" << endl;
    if (argc != 5) {
        cout << "Usage: " << endl;
        cout << argv[0] << " -src <input_folder> -dest <output_folder>" << endl;
        return 1;
    }

    // ********************
    // Get input parameters
    // ********************

    string input;
    string output_folder;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-src") == 0) {
            if (i + 1 >= argc) return -1;
            input = argv[i + 1];
        }

        if (strcmp(argv[i], "-dest") == 0) {
            if (i + 1 >= argc) return -1;
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
    vector<string> imgPath = listFile(input);

    int num_of_image = imgPath.size();

    int EYE_IMG_WIDTH = 100;
    int EYE_IMG_HEIGHT = 100;
    int MOUTH_IMG_WIDTH = 60;
    int MOUTH_IMG_HEIGHT = 40;
    int FACE_IMG_WIDTH = 160;
    int FACE_IMG_HEIGHT = 160;

    // ------ load cascade files ----
    CascadeClassifier face_cascade = loadCascade("haarcascade_frontalface_alt.xml");
    if (face_cascade.empty()) return;

    FLANDMARK_Model *model = flandmark_init("flandmark_model.dat");
    int num_of_landmark = model->data.options.M;
    double *points = new double[2 * num_of_landmark];

    FileStorage fs(output + "/list.yml", FileStorage::WRITE);
    fs << "num_of_image" << num_of_image;

    int real_img_num = 0;
    for (int img_id = 0; img_id < num_of_image; img_id++) {
        Mat img, img_gray;
        // load image

//        cout << "ImagePath: " << imgPath[img_id] << endl;

        img = imread(imgPath[img_id], CV_LOAD_IMAGE_COLOR);

//        cout<<img.elemSize1()<<endl;

//        cout<<img.type()<<endl;
        //16

        cvtColor(img, img_gray, CV_RGB2GRAY);
//        img_gray = img;
        equalizeHist(img_gray, img_gray);

//        Mat imgtemp = img_gray;

//        cout << "1" << endl;

        vector<Rect> faces;
        face_cascade.detectMultiScale(img_gray, faces, 1.1, 3);

//        cout << faces.size() << endl;

        if (faces.size() != 0) {

            // Get the largest face region
            int i = 0;
            int max_width = 0;
            for (int index = 0; index < faces.size(); index++) {
//                cout << "for in" << endl;
                if (faces[i].width > max_width) {
                    i = index;
                    max_width = faces[i].width;
                }
            }

//            cout << "2" << endl;
//            cout << "i : " << i << endl;
//            cout << faces.size() << endl;
//        if(faces.size() != 0)


//            cout << "sieze : " << faces[i].size() << endl;
//            cout << "faces[i].x : " << faces[i].x << endl;
//            cout << "faces[i].y : " << faces[i].y << endl;
//            cout << "faces[i].x + faces[i].width : " << faces[i].x + faces[i].width << endl;
//            cout << "faces[i].y + faces[i].height : " << faces[i].y + faces[i].height << endl;

//            if (!(faces[i].width < (img_gray.cols / 3) || faces[i].height < (img_gray.cols / 3))) {

            int bbox[4] = {faces[i].x, faces[i].y, faces[i].x + faces[i].width, faces[i].y + faces[i].height};

//                cout << "211" << endl;
            flandmark_detect(new IplImage(img_gray), bbox, model, points);

//                cout << "21" << endl;
            // left eye
            Point centerLeft = Point((int) (points[2 * 6] + points[2 * 2]) / 2,
                                     (int) (points[2 * 6 + 1] + points[2 * 2 + 1]) / 2);
            int widthLeft = abs(points[2 * 6] - points[2 * 2]);

//                cout << "22" << endl;
            // right eye
            Point centerRight = Point((int) (points[2 * 1] + points[2 * 5]) / 2,
                                      (int) (points[2 * 1 + 1] + points[2 * 5 + 1]) / 2);
            int widthRight = abs(points[2 * 1] - points[2 * 5]);

//                cout << "23" << endl;
//
//                cout << "centerLeft.x + widthLeft : " << centerLeft.x + widthLeft << endl;
//                cout << "centerRight.x - widthRight : " << centerRight.x - widthRight << endl;
            // face
//            int widthFace = (centerLeft.x + widthLeft) - (centerRight.x - widthRight) + 15;
            int widthFace = (centerLeft.x + widthLeft) - (centerRight.x - widthRight);

            int heightFace = widthFace * 1.3;

//            if (0 < (centerRight.x - widthFace / 4) + widthFace &&
//                (centerRight.x - widthFace / 4) + widthFace < 255 &&
//                0 < (centerRight.y - heightFace / 4) + heightFace &&
//                (centerRight.y - heightFace / 4) + heightFace < 255 &&
//                widthFace > 140 && heightFace > 170) {

//
//                    cout << "centerRight.x - widthFace/4 : " << centerRight.x - widthFace / 4 << endl;
//                    cout << "centerRight.y - heightFace/4 : " << centerRight.y - heightFace / 4 << endl;
//                    cout << "widthFace : " << widthFace << endl;
//                    cout << "heightFace : " << heightFace << endl;

                Mat face = img(
                        Rect(centerRight.x - widthFace / 4, centerRight.y - heightFace / 4, widthFace, heightFace));


//                cout<<"face type : "<<face.type()<<endl;
//                cout<<"face ele : "<<face.elemSize1()<<endl;

//                face = imgtemp;
//                    cout << "3" << endl;
                //
                // extract label
//                cout << "ImagePath: " << imgPath[img_id] << endl;
                string fileName = imgPath[img_id].substr(input.length() + 1, imgPath[i].length());

                // save image
                string curFileName = fileName;

                curFileName = fileName;


                //        curFileName.replace(fileName.length() - 4, 4, "face.tiff");
//                    curFileName.replace(fileName.length() - 4, 4, "face.png");

//                    cout << "4" << endl;
//
//                    cout << "FACE_IMG_WIDTH : " << FACE_IMG_WIDTH << endl;
//                    cout << "FACE_IMG_HEIGHT : " << FACE_IMG_HEIGHT << endl;
//                    cout << face.size << endl;
                resize(face, face, Size(FACE_IMG_WIDTH, FACE_IMG_HEIGHT));

                if(curFileName.substr(2,1) == ".")
                    curFileName.replace(fileName.length() - 4, 4, "face.png");

                imwrite(output + "/" + curFileName, face);
                fs << "img_" + to_string(real_img_num) + "_face" << output + "/" + curFileName;


                cout << "real img num : " << real_img_num << endl;
                real_img_num++;
//                } else {
//                    cout << "image's bbox scale is too small" << endl;
//                }
//            } else {
//                cout << "image's flandmark_detect Error" << endl;
//            }
        } else {
            cout << "image's detectMultiScale Error" << endl;
        }

//        cout << "img num : " << img_id << endl;

//        cout << "Next For ===================================================" << endl;
    }

    fs << "real_num_of_image" << real_img_num;

}

CascadeClassifier loadCascade(string cascadePath) {
    CascadeClassifier cascade;
    if (!cascade.load(cascadePath)) {
        cout << "--(!)Error loading cascade: " << cascadePath << endl;
    };
    return cascade;
}

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