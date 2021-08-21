//
// Created by bismarck on 2021/6/14.
//
#include "retinaface/hair.h"
#include <vector>
#include <cmath>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

const int Cx = 116, Cy = 138;
const int Ecx = 172, Ecy = 170;
const float a = 25.39, b = 14.03;
const float Theta = 2.53;
const float thresh = 0.5;

void PerfectReflectionAlgorithm(Mat src, Mat &dst) {
    int row = src.rows;
    int col = src.cols;
    dst = Mat(row, col, CV_8UC3);
    int HistRGB[767] = { 0 };
    int MaxVal = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[0]);
            MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[1]);
            MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[2]);
            int sum = src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2];
            HistRGB[sum]++;
        }
    }
    int Threshold = 0;
    int sum = 0;
    for (int i = 766; i >= 0; i--) {
        sum += HistRGB[i];
        if (sum > row * col * 0.1) {
            Threshold = i;
            break;
        }
    }
    int AvgB = 0;
    int AvgG = 0;
    int AvgR = 0;
    int cnt = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            int sumP = src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2];
            if (sumP > Threshold) {
                AvgB += src.at<Vec3b>(i, j)[0];
                AvgG += src.at<Vec3b>(i, j)[1];
                AvgR += src.at<Vec3b>(i, j)[2];
                cnt++;
            }
        }
    }
    AvgB /= cnt;
    AvgG /= cnt;
    AvgR /= cnt;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            int Blue = src.at<Vec3b>(i, j)[0] * MaxVal / AvgB;
            int Green = src.at<Vec3b>(i, j)[1] * MaxVal / AvgG;
            int Red = src.at<Vec3b>(i, j)[2] * MaxVal / AvgR;
            if (Red > 255) {
                Red = 255;
            }
            else if (Red < 0) {
                Red = 0;
            }
            if (Green > 255) {
                Green = 255;
            }
            else if (Green < 0) {
                Green = 0;
            }
            if (Blue > 255) {
                Blue = 255;
            }
            else if (Blue < 0) {
                Blue = 0;
            }
            dst.at<Vec3b>(i, j)[0] = Blue;
            dst.at<Vec3b>(i, j)[1] = Green;
            dst.at<Vec3b>(i, j)[2] = Red;
        }
    }
}

void AutoLevelsAdjust(const cv::Mat &src1, cv::Mat &dst) {
    Mat src = src1.clone();
    CV_Assert(!src.empty() && src.channels() == 3);
    int BHist[256] = {0};
    int GHist[256] = {0};
    int RHist[256] = {0};
    cv::MatIterator_<Vec3b> its, ends;
    for (its = src.begin<Vec3b>(), ends = src.end<Vec3b>(); its != ends; its++) {
        BHist[(*its)[0]]++;
        GHist[(*its)[1]]++;
        RHist[(*its)[2]]++;
    }

    float LowCut = 0.5;
    float HighCut = 0.5;

    int BMax = 0, BMin = 0;
    int GMax = 0, GMin = 0;
    int RMax = 0, RMin = 0;

    int TotalPixels = src.cols * src.rows;
    float LowTh = LowCut * 0.01 * TotalPixels;
    float HighTh = HighCut * 0.01 * TotalPixels;

    int sumTempB = 0;
    for (int i = 0; i < 256; i++) {
        sumTempB += BHist[i];
        if (sumTempB >= LowTh) {
            BMin = i;
            break;
        }
    }
    sumTempB = 0;
    for (int i = 255; i >= 0; i--) {
        sumTempB += BHist[i];
        if (sumTempB >= HighTh) {
            BMax = i;
            break;
        }
    }

    int sumTempG = 0;
    for (int i = 0; i < 256; i++) {
        sumTempG += GHist[i];
        if (sumTempG >= LowTh) {
            GMin = i;
            break;
        }
    }
    sumTempG = 0;
    for (int i = 255; i >= 0; i--) {
        sumTempG += GHist[i];
        if (sumTempG >= HighTh) {
            GMax = i;
            break;
        }
    }

    int sumTempR = 0;
    for (int i = 0; i < 256; i++) {
        sumTempR += RHist[i];
        if (sumTempR >= LowTh) {
            RMin = i;
            break;
        }
    }
    sumTempR = 0;
    for (int i = 255; i >= 0; i--) {
        sumTempR += RHist[i];
        if (sumTempR >= HighTh) {
            RMax = i;
            break;
        }
    }

    int BTable[256] = {0};
    for (int i = 0; i < 256; i++) {
        if (i <= BMin)
            BTable[i] = 0;
        else if (i > BMin && i < BMax)
            BTable[i] = cvRound((float) (i - BMin) / (BMax - BMin) * 255);
        else
            BTable[i] = 255;
    }

    int GTable[256] = {0};
    for (int i = 0; i < 256; i++) {
        if (i <= GMin)
            GTable[i] = 0;
        else if (i > GMin && i < GMax)
            GTable[i] = cvRound((float) (i - GMin) / (GMax - GMin) * 255);
        else
            GTable[i] = 255;
    }

    int RTable[256] = {0};
    for (int i = 0; i < 256; i++) {
        if (i <= RMin)
            RTable[i] = 0;
        else if (i > RMin && i < RMax)
            RTable[i] = cvRound((float) (i - RMin) / (RMax - RMin) * 255);
        else
            RTable[i] = 255;
    }

    cv::Mat dst_ = src.clone();
    cv::MatIterator_<Vec3b> itd, endd;
    for (itd = dst_.begin<Vec3b>(), endd = dst_.end<Vec3b>(); itd != endd; itd++) {
        (*itd)[0] = BTable[(*itd)[0]];
        (*itd)[1] = GTable[(*itd)[1]];
        (*itd)[2] = RTable[(*itd)[2]];
    }
    dst = dst_;
}

double wash(const Mat &_image) {
    Mat image;
    bitwise_not(_image, image);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    double aero = 0;
    findContours(image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
    for (auto &contour : contours) {
        bool ok = false;
        for (auto &point : contour) {
            if (point.y < image.rows / 2) {
                ok = true;
                break;
            }
        }
        if (ok) {
            double aero_t = contourArea(contour) / image.rows / image.cols;
            if (aero_t > aero) {
                aero = aero_t;
            }
        }
    }
    return aero;
}

double getHair(const Mat &_input) {
    Mat input, out;
    PerfectReflectionAlgorithm(_input, out);
    AutoLevelsAdjust(out, input);
    int m = input.rows;
    int n = input.cols;
    Mat bw = Mat::zeros(m, n, CV_8UC1);

    Mat YCbCr, Y, Cr, Cb;
    vector<Mat> channels;
    cvtColor(input, YCbCr, COLOR_BGR2YCrCb);
    split(YCbCr, channels);
    Y = channels.at(0);
    Cr = channels.at(1);
    Cb = channels.at(2);

    Y.convertTo(Y, CV_32FC1);
    Cr.convertTo(Cr, CV_32FC1);
    Cb.convertTo(Cb, CV_32FC1);

    Mat RotateM = (Mat_<float>(2, 2) <<
                                     cos(Theta), sin(Theta), -sin(Theta), cos(Theta));
    Mat Diff = Mat::zeros(2, 1, CV_32FC1);;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            Diff.at<float>(0, 0) = Cb.at<float>(i, j) - Cx;
            Diff.at<float>(1, 0) = Cr.at<float>(i, j) - Cy;
            Mat RotateVal = RotateM * Diff;
            float x = RotateVal.at<float>(0, 0);
            float y = RotateVal.at<float>(1, 0);
            float EllipseV = pow((x - Ecx / 100), 2.0) / (a * a) + pow((y - Ecy / 100), 2.0) / (b * b);
            if (EllipseV <= 1) {
                bw.at<uchar>(i, j) = 255;
            }
            if (Y.at<float>(i, j) < 80.0) {
                bw.at<uchar>(i, j) = 0;
            }
        }
    }
    //imshow("原始肤色区域", SkinBW);
    int KnelW = 3;
    morphologyEx(bw, bw, MORPH_OPEN, Mat(KnelW, KnelW, CV_8U), Point(-1, -1), 1);\
//    imshow("raw", input);
//    imshow("b", bw);
//    waitKey(0);
    double aero = wash(bw);
    return aero;
}