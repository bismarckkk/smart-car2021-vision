//
// Created by bismarck on 2021/7/19.
//

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "retinaface5/glass.h"

using namespace cv;
using namespace std;

void PerfectReflectionAlgorithm2(Mat src, Mat &dst) {
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

void AutoLevelsAdjust2(const cv::Mat &src1, cv::Mat &dst) {
    Mat src = src1.clone();
    CV_Assert(!src.empty() && src.channels() == 3);
    //统计灰度直方图
    int BHist[256] = {0};    //B分离
    int GHist[256] = {0};    //G分量
    int RHist[256] = {0};    //R分量
    cv::MatIterator_<Vec3b> its, ends;
    for (its = src.begin<Vec3b>(), ends = src.end<Vec3b>(); its != ends; its++) {
        BHist[(*its)[0]]++;
        GHist[(*its)[1]]++;
        RHist[(*its)[2]]++;
    }

    //设置LowCut和HighCut
    float LowCut = 0.5;
    float HighCut = 0.5;

    //根据LowCut和HighCut查找每个通道最大值最小值
    int BMax = 0, BMin = 0;
    int GMax = 0, GMin = 0;
    int RMax = 0, RMin = 0;

    int TotalPixels = src.cols * src.rows;
    float LowTh = LowCut * 0.01 * TotalPixels;
    float HighTh = HighCut * 0.01 * TotalPixels;

    //B通道查找最小最大值
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

    //G通道查找最小最大值
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

    //R通道查找最小最大值
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

    //对每个通道建立分段线性查找表
    //B分量查找表
    int BTable[256] = {0};
    for (int i = 0; i < 256; i++) {
        if (i <= BMin)
            BTable[i] = 0;
        else if (i > BMin && i < BMax)
            BTable[i] = cvRound((float) (i - BMin) / (BMax - BMin) * 255);
        else
            BTable[i] = 255;
    }

    //G分量查找表
    int GTable[256] = {0};
    for (int i = 0; i < 256; i++) {
        if (i <= GMin)
            GTable[i] = 0;
        else if (i > GMin && i < GMax)
            GTable[i] = cvRound((float) (i - GMin) / (GMax - GMin) * 255);
        else
            GTable[i] = 255;
    }

    //R分量查找表
    int RTable[256] = {0};
    for (int i = 0; i < 256; i++) {
        if (i <= RMin)
            RTable[i] = 0;
        else if (i > RMin && i < RMax)
            RTable[i] = cvRound((float) (i - RMin) / (RMax - RMin) * 255);
        else
            RTable[i] = 255;
    }

    //对每个通道用相应的查找表进行分段线性拉伸
    cv::Mat dst_ = src.clone();
    cv::MatIterator_<Vec3b> itd, endd;
    for (itd = dst_.begin<Vec3b>(), endd = dst_.end<Vec3b>(); itd != endd; itd++) {
        (*itd)[0] = BTable[(*itd)[0]];
        (*itd)[1] = GTable[(*itd)[1]];
        (*itd)[2] = RTable[(*itd)[2]];
    }
    dst = dst_;
}


int res;
Mat _input;

vector<Mat> examples;

void _getHair(int, void*) {
    resize(_input, _input, Size(145, 135));
    auto match_method = TM_SQDIFF_NORMED;
    cout << "classing\n";
    vector<double> minValue;
    double min_long_hair = 10, min_short_hair = 10, min_glass = 10, min_no_glass = 10;
    vector<int> long_hair = {1, 5, 6, 7}, short_hair = {0, 2, 3, 4};
    vector<int> glass = {0, 1, 2}, no_glass = {3, 4, 5, 6, 7};
    for (const auto& templ : examples) {
        int result_cols =  templ.cols - _input.cols + 1;
        int result_rows = templ.rows - _input.rows + 1;
        Mat result( result_cols, result_rows, CV_32FC1 );
        /// 进行匹配和标准化
        matchTemplate( templ, _input, result, match_method );
        //normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

        /// 通过函数 minMaxLoc 定位最匹配的位置
        double minVal; double maxVal; Point minLoc; Point maxLoc;
        Point matchLoc;

        minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
        cout << minVal << endl;
        //matchLoc = minLoc;
        //rectangle(templ, matchLoc, Point(matchLoc.x + _input.cols, matchLoc.y + _input.rows), Scalar(0, 255, 0), 2, 8, 0);
        //imshow("0", templ);
        //waitKey(0);

        minValue.push_back(minVal);
    }
    double minVal = minValue[0];
    int id = 0;
    for (int i = 1; i < 8; i++) {
        if (minValue[i] < minVal) {
            minVal = minValue[i];
            id = i;
        }
    }
    cout << "It maybe " << id <<".jpg\n";
    res = 0;
    if (id == 1 || (id >= 5 && id <= 7)) {
        res += 10;
    }
    if (id >= 3 && id <= 7) {
        res += 1;
    }
    cout << "return " << res << endl;
}

int getGlass(const Mat &input) {
    Mat tmp;
    PerfectReflectionAlgorithm2(input, tmp);
    AutoLevelsAdjust2(tmp, _input);
    _getHair(0, NULL);
    //waitKey(0);
    return res;
}

void init_glass_detect() {
    Mat tmp;
    for (int i = 0; i < 8; i++) {
        tmp = imread(string (PACK_PATH) + "/example/" + to_string(i) + ".jpg");
        resize(tmp, tmp, Size(356, 536));
        examples.push_back(tmp);
    }
}

