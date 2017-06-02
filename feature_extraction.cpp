#include <iostream>

#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;

Mat extractFeature(Mat face, string feature_name);
void createDenseFeature(vector<KeyPoint> &keypoints, Mat image, float initFeatureScale=1.f, int featureScaleLevels=1,
                                    float featureScaleMul=0.1f,
                                    int initXyStep=6, int initImgBound=0,
                                    bool varyXyStepWithScale=true,
                                    bool varyImgBoundWithScale=false);
int main(int argc, char* argv[]) {
    cout << "============= FEATURE EXTRACTION =============" << endl;

    if(argc != 7){
        cout << "Usage: " << endl;
        cout << argv[0] << " -feature <feature_name> -src <input_folder> -dest <output_folder>" << endl;
        return 1;
    }

    // ********************
    // Get input parameters
    // ********************

    string feature_name;
    string input_folder;
    string output_folder;

    for(int i = 0; i < argc ; i++)
    {
        if( strcmp(argv[i], "-feature") == 0 ){
            if(i + 1 >= argc) return -1;
            feature_name = argv[i + 1];
        }

        if( strcmp(argv[i], "-src") == 0 ){
            if(i + 1 >= argc) return -1;
            input_folder = argv[i + 1];
        }

        if( strcmp(argv[i], "-dest") == 0 ){
            if(i + 1 >= argc) return -1;
            output_folder = argv[i + 1];
        }
    }

    // *********************
    // extract feature
    // *********************

    cout << "feature: " << feature_name << endl;
    cout << "src: " << input_folder << endl;
    cout << "dest: " << output_folder << endl;

    // READ YML FILES
    string input_file = input_folder + "/list.yml";
    FileStorage in( input_file, FileStorage::READ);

    string output_path = output_folder + "/" + feature_name +"_features.yml";
    FileStorage fs( output_path , FileStorage::WRITE);

    int num_of_image = 0;
    in["num_of_image"] >> num_of_image;


    int real_num_of_image = 0;
    in["real_num_of_image"] >> real_num_of_image;


//    cout << "num of image : " << num_of_image << endl;
    cout << "real_num_of_image : " << real_num_of_image << endl;
    num_of_image = real_num_of_image;

    vector<Mat> features_vector;

    // extract image features from facial components
    for(int i = 0 ; i < num_of_image; i++){

        string facePath;
        in["img_" + to_string(i) + "_face"] >> facePath;

        Mat face = imread(facePath, CV_LOAD_IMAGE_GRAYSCALE);
        resize(face, face, Size(80, 80));

        /////
//        face.convertTo(face,CV_8UC1);
//        cvtColor(face, face, CV_GRAY2RGB);
        //////

//        cout<<"face.col"<<face.cols<<endl<<endl<<endl<<endl<<endl;

        Mat result = extractFeature(face, feature_name);


        features_vector.push_back(result);

//        cout<<"result.col"<<result.cols<<endl<<endl<<endl<<endl<<endl;
    }


    // --------------- clustering -----------------
    // compute feature distribution over k bins
    // create 1 featureData Mat with each feature as a row
    int num_of_feature = 0;
    for(int i = 0 ; i < features_vector.size(); i++){
        num_of_feature += features_vector[i].rows;
        cout<<"features_vector[i].rows : "<<features_vector[i].rows<<endl;
    }
    cout << "num_of_feature: " << num_of_feature << endl;

    cout<<"features_vector[0].cols : "<<features_vector[0].cols<<endl;

    Mat rawFeatureData = Mat::zeros(num_of_feature, features_vector[0].cols, CV_32FC1);
    int cur_idx = 0;

    for(int i = 0 ; i < features_vector.size(); i++){
        features_vector[i].copyTo(rawFeatureData.rowRange(cur_idx, cur_idx + features_vector[i].rows));
        cur_idx += features_vector[i].rows;
    }

    // --- compute kmeans
    Mat labels, centers;

    cout<<"label : "<<endl;
    cout<<labels<<endl;

//    cout<<"rawFeaturedata raw: "<<rawFeatureData<<endl;

    cout<<"rawFeaturedata : "<<rawFeatureData.row(0)<<endl;


    int bin_size = 256;
//    kmeans(rawFeatureData, bin_size, labels, TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 100, 1.0),
//           3, KMEANS_PP_CENTERS, centers);
//    cout<<"rawFeaturedata : "<<rawFeatureData<<endl;


    cout<<"label : "<<endl;
//    cout<<labels<<endl;
//    cout<<labels<<endl;


    ////N이 K보다 커야한다

//    CV_EXPORTS_W double kmeans( InputArray data, int K, InputOutputArray bestLabels,
//                                TermCriteria criteria, int attempts,
//                                int flags, OutputArray centers = noArray() );


//    kmeans(values, K, labels, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 1.0), 10, KMEANS_PP_CENTERS, centers)

//    kmeans(cv::Mat(values).reshape(1, values.size()), K, labels, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 1.0), 10, KMEANS_PP_CENTERS, centers)


    //label -> label -> bin -> feature -> featureDataOverBins -> feature

    // compute final feature
    cur_idx = 0;
    int num_of_test = 0;
    RNG rng;

    Mat featureDataOverBins = Mat::zeros(num_of_image, bin_size, CV_32FC1); //[num_of_image * bin_size]
    string fileName;

    cout<<"bin_size : "<<bin_size<<endl;

    cout<<rawFeatureData.type()<<endl;
    for(int i = 0 ; i < num_of_image; i++)
    {
        // for each image
        Mat feature = Mat::zeros(1, bin_size, CV_32FC1);
        int num_of_image_features = features_vector[i].rows;
        for(int j = 0; j < num_of_image_features; j++){ // for each feature in cur image
            for(int k = 0; k < 128; k++){ // for each feature in cur image

//                    int bin = labels.at<int>(cur_idx + j);
                    int bin = (int) rawFeatureData.at<float>(cur_idx + j, k);
                cout<<bin<<endl;
                    feature.at<float>(0, bin) += 1;
                cout<<"k : "<<k<<endl;
                }
            cout<<"j : "<<j<<endl;
        }

        cout<<"first feature :"<<endl<<feature<<endl;


        normalize( feature, feature );




        cout<<"first feature after nomal:"<<endl<<feature<<endl;


        string path;
        in["img_" + to_string(i) + "_face"] >> path;

        // extract label
        fileName = path.substr(input_folder.length() + 1, path.length());

        string ex_code;
        int label;
        if(fileName.substr(2,1) == ".") {

            ex_code= fileName.substr(3, 2);
            label = -1;
            if (ex_code == "AN") {
                label = 0;
            } else if (ex_code == "DI") {
                label = 1;
            } else if (ex_code == "FE") {
                label = 2;
            } else if (ex_code == "HA") {
                label = 3;
            } else if (ex_code == "NE") {
                label = 4;
            } else if (ex_code == "SA") {
                label = 5;
            } else if (ex_code == "SU") {
                label = 6;
            }
        } else {

            ex_code = fileName.substr(2, 1);
            label = -1;
            if (ex_code == "0") {
                label = 0;
            } else if (ex_code == "1") {
                label = 1;
            } else if (ex_code == "2") {
                label = 2;
            } else if (ex_code == "3") {
                label = 3;
            } else if (ex_code == "6") {
                label = 4;
            } else if (ex_code == "4") {
                label = 5;
            } else if (ex_code == "5") {
                label = 6;
            }

        }

//        cout << "histogram feature: " << endl << feature << " label: " << label << " " <<  path << endl;

//        fs << "image_feature_" + to_string(i) << feature;


        fs << "image_label_" + to_string(i) << label;

        cout<<"Asdf"<<endl;

        // decide train or test
        double c = rng.uniform(0., 1.);
        bool isTrain = true;
        if(c>0.9){
            isTrain = false;
            num_of_test += 1;
        }
        fs << "image_is_train_" + to_string(i) << isTrain;

        feature.copyTo(featureDataOverBins.row(i));

        cur_idx += features_vector[i].rows;


        cout<<"i : "<<i<<endl;
    }


    cout<<"featureDataOverBins.row(0) : "<<endl;
//    cout<<featureDataOverBins<<endl;

    cout<<"featureDataOverBins.rows : "<<featureDataOverBins.rows<<endl;
    cout<<"featureDataOverBins.cols : "<<featureDataOverBins.cols<<endl;



//    // --- PCA
//    // perform PCA
//    PCA pca(featureDataOverBins, cv::Mat(), CV_PCA_DATA_AS_ROW);

//    PCA pca(featureDataOverBins, cv::Mat(), CV_PCA_DATA_AS_ROW, 0.9);

//    int feature_size = pca.eigenvectors.rows;

//    cout<<" pca.eigenvectors.cols; :"<< pca.eigenvectors.cols<<endl;

//    cout<<" pca.eigenvectors.size :"<< pca.eigenvectors.size<<endl;
//    cout<<"featureDataOverBins : "<<featureDataOverBins<<endl;

    Mat feature;

    int datnum = 0;
    for(int i = 0 ; i < num_of_image; i++){
//        feature = pca.project(featureDataOverBins.row(i));
        feature = featureDataOverBins.row(i);

//        normalize(feature, feature, 0, 0.1, NORM_MINMAX, CV_32F);

        fs << "image_feature_" + to_string(i) << feature;
        if(i<4)
        cout<<"featureDataOverBins.row(i) : "<<endl<<featureDataOverBins.row(i)<<endl;
//        cout<<"feature : "<<endl<<feature<<endl;
    }
//    cout << "feature_size: " << feature_size << endl;

    cout<<"feaeture : "<<feature.size<<endl;

//    fs << "feature_size" << feature_size;
    // save result

    fs << "num_of_image" << num_of_image;
    fs << "num_of_label" << 7;
    if(fileName.substr(2,1) == ".") {
        fs << "label_0" << "Angry";
        fs << "label_1" << "Disgusted";
        fs << "label_2" << "Fear";
        fs << "label_3" << "Happy";
        fs << "label_4" << "Neural";
        fs << "label_5" << "Sad";
        fs << "label_6" << "Surprised";

    }else{
        fs << "label_0" << "Angry";
        fs << "label_1" << "Disgusted";
        fs << "label_2" << "Fear";
        fs << "label_3" << "Happy";
        fs << "label_4" << "Sad";
        fs << "label_5" << "Suprised";
        fs << "label_6" << "Neural";
    }

    fs << "num_of_train" << num_of_image - num_of_test;
    fs << "num_of_test" << num_of_test;
//    fs << "pca_mean" << pca.mean;
//    fs << "pca_eigenvalues" << pca.eigenvalues;
//    fs << "pca_eigenvectors" << pca.eigenvectors;
    fs << "centers" << centers;

    fs.release();


    cout << "Save features: " << output_path << endl;
    return 0;
}

void visualizeKeypoints(Mat gray, vector<KeyPoint> keypoints, String feature_name){

    Mat img;
    cvtColor(gray, img, CV_GRAY2BGR);
    cout << "channels : " << img.channels() << endl;
    for(int i = 0 ; i < keypoints.size() ; i ++){
        circle(img, keypoints[i].pt, keypoints[i].size, (255, 255 , 255), 1);
    }

    resize(img, img , Size(200, 200));

    imshow("keypoint", img);
    waitKey(0);
}

Mat extractBrisk(Mat img){
    Mat descriptors;
    vector<KeyPoint> keypoints;

    Ptr<DescriptorExtractor> brisk = BRISK::create();
    brisk->detect(img, keypoints, Mat());
    brisk->compute(img, keypoints, descriptors);

    return descriptors;
}
Mat extractKaze(Mat img){
    Mat descriptors;
    vector<KeyPoint> keypoints;

    Ptr<DescriptorExtractor> kaze = KAZE::create();
    kaze->detect(img, keypoints, Mat());
    kaze->compute(img, keypoints, descriptors);

    return descriptors;
}
Mat extractSift(Mat img){
    Mat descriptors;
    vector<KeyPoint> keypoints;

    Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
    sift->detect(img, keypoints, Mat());
    sift->compute(img, keypoints, descriptors);

    return descriptors;
}
Mat extractSurf(Mat img){
    Mat descriptors;
    vector<KeyPoint> keypoints;

    Ptr<Feature2D> surf = xfeatures2d::SURF::create();
    surf->detect(img, keypoints, Mat());
    surf->compute(img, keypoints, descriptors);

    return descriptors;
}
Mat extractDenseSift(Mat img){
    Mat descriptors;
    vector<KeyPoint> keypoints;

    Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
    createDenseFeature(keypoints, img);

    sift->compute(img, keypoints, descriptors);

    return descriptors;
}
Mat extractDaisy(Mat img){
    Mat descriptors;
    vector<KeyPoint> keypoints;

    Ptr<FeatureDetector> surf = xfeatures2d::SURF::create();
    surf->detect(img, keypoints, Mat());
    Ptr<DescriptorExtractor> daisy = xfeatures2d::DAISY::create();
    daisy->compute(img, keypoints, descriptors);

    return descriptors;
}
Mat extractFeature(Mat img, string feature_name){
    Mat descriptors;
    if(feature_name.compare("brisk") == 0){
        descriptors = extractBrisk(img);
    } else if(feature_name.compare("kaze") == 0){
        descriptors = extractKaze(img);
    } else if(feature_name.compare("sift") == 0){
        descriptors = extractSift(img);
    } else if(feature_name.compare("surf") == 0){
        descriptors = extractSurf(img);
    } else if(feature_name.compare("dense-sift") == 0){
        descriptors = extractDenseSift(img);
    } else if(feature_name.compare("daisy") == 0){
        descriptors = extractDaisy(img);
    }
    return descriptors;
}

void createDenseFeature(vector<KeyPoint> &keypoints, Mat image, float initFeatureScale, int featureScaleLevels,
                                    float featureScaleMul,
                                    int initXyStep, int initImgBound,
                                    bool varyXyStepWithScale,
                                    bool varyImgBoundWithScale){
    float curScale = static_cast<float>(initFeatureScale);
    int curStep = initXyStep;
    int curBound = initImgBound;
    for( int curLevel = 0; curLevel < featureScaleLevels; curLevel++ )
    {
        for( int x = curBound; x < image.cols - curBound; x += curStep )
        {
            for( int y = curBound; y < image.rows - curBound; y += curStep )
            {
                keypoints.push_back( KeyPoint(static_cast<float>(x), static_cast<float>(y), curScale) );
            }
        }

        curScale = static_cast<float>(curScale * featureScaleMul);
        if( varyXyStepWithScale ) curStep = static_cast<int>( curStep * featureScaleMul + 0.5f );
        if( varyImgBoundWithScale ) curBound = static_cast<int>( curBound * featureScaleMul + 0.5f );
    }
}