#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "retinaface5/common.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <ros/ros.h>
#include "retinaface5/plugin_factory.h"
#include "retinaface5/decode.h"
#include "retinaface5/detect.h"
#include "retinaface5/glass.h"
#include <opencv2/opencv.hpp>

//#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id

// stuff we know about the network and the input/output blobs
static const int INPUT_H = decodeplugin::INPUT_H;  // H, W must be able to  be divided by 32.
static const int INPUT_W = decodeplugin::INPUT_W;;
static const int OUTPUT_SIZE = (INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 2  * 15 + 1;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
nvinfer1::IExecutionContext* context;
ros::Publisher res_pub;

using namespace nvinfer1;
static Logger gLogger;

cv::Mat preprocess_img(const cv::Mat& img) {
    int w, h, x, y;
    float r_w = INPUT_W / (img.cols*1.0);
    float r_h = INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    } else {
        w = r_h* img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

cv::Rect get_rect_adapt_landmark(const cv::Mat& img, float bbox[4], float lmk[10]) {
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] / r_w;
        r = bbox[2] / r_w;
        t = (bbox[1] - (INPUT_H - r_w * img.rows) / 2) / r_w;
        b = (bbox[3] - (INPUT_H - r_w * img.rows) / 2) / r_w;
        for (int i = 0; i < 10; i += 2) {
            lmk[i] /= r_w;
            lmk[i + 1] = (lmk[i + 1] - (INPUT_H - r_w * img.rows) / 2) / r_w;
        }
    } else {
        l = (bbox[0] - (INPUT_W - r_h * img.cols) / 2) / r_h;
        r = (bbox[2] - (INPUT_W - r_h * img.cols) / 2) / r_h;
        t = bbox[1] / r_h;
        b = bbox[3] / r_h;
        for (int i = 0; i < 10; i += 2) {
            lmk[i] = (lmk[i] - (INPUT_W - r_h * img.cols) / 2) / r_h;
            lmk[i + 1] /= r_h;
        }
    }
    int w = r-l, h = b-t;
    l -= w / 2;
    r += w / 2;
    t -= h / 2;
    b += h / 1.5;
    if (l < 0) {
        l = 0;
    }
    if (t < 0) {
        t = 0;
    }
    if (r > img.cols) {
        r = img.cols;
    }
    if (b > img.rows) {
        b = img.rows;
    }
    return cv::Rect(l, t, r-l, b-t);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        max(lbox[0], rbox[0]), //left
        min(lbox[2], rbox[2]), //right
        max(lbox[1], rbox[1]), //top
        min(lbox[3], rbox[3]), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) -interBoxS + 0.000001f);
}

bool cmp(decodeplugin::Detection& a, decodeplugin::Detection& b) {
    return a.class_confidence > b.class_confidence;
}

void nms(std::vector<decodeplugin::Detection>& res, float *output, float nms_thresh = 0.4) {
    std::vector<decodeplugin::Detection> dets;
    for (int i = 0; i < output[0]; i++) {
        if (output[15 * i + 1 + 4] <= 0.1) continue;
        decodeplugin::Detection det;
        memcpy(&det, &output[15 * i + 1], sizeof(decodeplugin::Detection));
        dets.push_back(det);
    }
    std::sort(dets.begin(), dets.end(), cmp);
    if (dets.size() > 5000) dets.erase(dets.begin() + 5000, dets.end());
    for (size_t m = 0; m < dets.size(); ++m) {
        auto& item = dets[m];
        res.push_back(item);
        //std::cout << item.class_confidence << " bbox " << item.bbox[0] << ", " << item.bbox[1] << ", " << item.bbox[2] << ", " << item.bbox[3] << std::endl;
        for (size_t n = m + 1; n < dets.size(); ++n) {
            if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                dets.erase(dets.begin()+n);
                --n;
            }
        }
    }
}

// Load weights from files
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

Weights getWeights(std::map<std::string, Weights>& weightMap, std::string key) {
    if (weightMap.count(key) != 1) {
        std::cerr << key << " not existed in weight map, fatal error!!!" << std::endl;
        exit(-1);
    }
    return weightMap[key];
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IActivationLayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{1, 1}, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStride(DimsHW{stride, stride});
    conv2->setPadding(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* conv3 = network->addConvolution(*relu2->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[lname + "conv3.weight"], emptywts);
    assert(conv3);

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);

    IElementWiseLayer* ew1;
    if (stride != 1 || inch != outch * 4) {
        IConvolutionLayer* conv4 = network->addConvolution(input, outch * 4, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv4);
        conv4->setStride(DimsHW{stride, stride});

        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}

ILayer* conv_bn_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int kernelsize, int stride, int padding, bool userelu, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{kernelsize, kernelsize}, getWeights(weightMap, lname + ".0.weight"), emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{stride, stride});
    conv1->setPadding(DimsHW{padding, padding});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);

    if (!userelu) return bn1;

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    return relu1;
}

IActivationLayer* ssh(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname) {
    auto conv3x3 = conv_bn_relu(network, weightMap, input, 256 / 2, 3, 1, 1, false, lname + ".conv3X3");
    auto conv5x5_1 = conv_bn_relu(network, weightMap, input, 256 / 4, 3, 1, 1, true, lname + ".conv5X5_1");
    auto conv5x5 = conv_bn_relu(network, weightMap, *conv5x5_1->getOutput(0), 256 / 4, 3, 1, 1, false, lname + ".conv5X5_2");
    auto conv7x7 = conv_bn_relu(network, weightMap, *conv5x5_1->getOutput(0), 256 / 4, 3, 1, 1, true, lname + ".conv7X7_2");
    conv7x7 = conv_bn_relu(network, weightMap, *conv7x7->getOutput(0), 256 / 4, 3, 1, 1, false, lname + ".conv7x7_3");
    ITensor* inputTensors[] = {conv3x3->getOutput(0), conv5x5->getOutput(0), conv7x7->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 3);
    IActivationLayer* relu1 = network->addActivation(*cat->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    return relu1;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, DataType dt) {
    INetworkDefinition* network = builder->createNetwork();

    // Create input tensor of shape { 1, 1, 32, 32 } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../retinaface.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // ------------- backbone resnet50 ---------------
    IConvolutionLayer* conv1 = network->addConvolution(*data, 64, DimsHW{7, 7}, weightMap["body.conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{2, 2});
    conv1->setPadding(DimsHW{3, 3});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "body.bn1", 1e-5);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPooling(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStride(DimsHW{2, 2});
    pool1->setPadding(DimsHW{1, 1});

    IActivationLayer* x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "body.layer1.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "body.layer1.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "body.layer1.2.");

    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 128, 2, "body.layer2.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.3.");
    IActivationLayer* layer2 = x;

    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 256, 2, "body.layer3.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.3.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.4.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.5.");
    IActivationLayer* layer3 = x;

    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 512, 2, "body.layer4.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "body.layer4.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "body.layer4.2.");
    IActivationLayer* layer4 = x;

    // ------------- FPN ---------------
    auto output1 = conv_bn_relu(network, weightMap, *layer2->getOutput(0), 256, 1, 1, 0, true, "fpn.output1");
    auto output2 = conv_bn_relu(network, weightMap, *layer3->getOutput(0), 256, 1, 1, 0, true, "fpn.output2");
    auto output3 = conv_bn_relu(network, weightMap, *layer4->getOutput(0), 256, 1, 1, 0, true, "fpn.output3");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer* up3 = network->addDeconvolution(*output3->getOutput(0), 256, DimsHW{2, 2}, deconvwts, emptywts);
    assert(up3);
    up3->setStride(DimsHW{2, 2});
    up3->setNbGroups(256);
    weightMap["up3"] = deconvwts;

    output2 = network->addElementWise(*output2->getOutput(0), *up3->getOutput(0), ElementWiseOperation::kSUM);
    output2 = conv_bn_relu(network, weightMap, *output2->getOutput(0), 256, 3, 1, 1, true, "fpn.merge2");

    IDeconvolutionLayer* up2 = network->addDeconvolution(*output2->getOutput(0), 256, DimsHW{2, 2}, deconvwts, emptywts);
    assert(up2);
    up2->setStride(DimsHW{2, 2});
    up2->setNbGroups(256);
    output1 = network->addElementWise(*output1->getOutput(0), *up2->getOutput(0), ElementWiseOperation::kSUM);
    output1 = conv_bn_relu(network, weightMap, *output1->getOutput(0), 256, 3, 1, 1, true, "fpn.merge1");

    // ------------- SSH ---------------
    auto ssh1 = ssh(network, weightMap, *output1->getOutput(0), "ssh1");
    auto ssh2 = ssh(network, weightMap, *output2->getOutput(0), "ssh2");
    auto ssh3 = ssh(network, weightMap, *output3->getOutput(0), "ssh3");

    // ------------- Head ---------------
    auto bbox_head1 = network->addConvolution(*ssh1->getOutput(0), 2 * 4, DimsHW{1, 1}, weightMap["BboxHead.0.conv1x1.weight"], weightMap["BboxHead.0.conv1x1.bias"]);
    auto bbox_head2 = network->addConvolution(*ssh2->getOutput(0), 2 * 4, DimsHW{1, 1}, weightMap["BboxHead.1.conv1x1.weight"], weightMap["BboxHead.1.conv1x1.bias"]);
    auto bbox_head3 = network->addConvolution(*ssh3->getOutput(0), 2 * 4, DimsHW{1, 1}, weightMap["BboxHead.2.conv1x1.weight"], weightMap["BboxHead.2.conv1x1.bias"]);

    auto cls_head1 = network->addConvolution(*ssh1->getOutput(0), 2 * 2, DimsHW{1, 1}, weightMap["ClassHead.0.conv1x1.weight"], weightMap["ClassHead.0.conv1x1.bias"]);
    auto cls_head2 = network->addConvolution(*ssh2->getOutput(0), 2 * 2, DimsHW{1, 1}, weightMap["ClassHead.1.conv1x1.weight"], weightMap["ClassHead.1.conv1x1.bias"]);
    auto cls_head3 = network->addConvolution(*ssh3->getOutput(0), 2 * 2, DimsHW{1, 1}, weightMap["ClassHead.2.conv1x1.weight"], weightMap["ClassHead.2.conv1x1.bias"]);

    auto lmk_head1 = network->addConvolution(*ssh1->getOutput(0), 2 * 10, DimsHW{1, 1}, weightMap["LandmarkHead.0.conv1x1.weight"], weightMap["LandmarkHead.0.conv1x1.bias"]);
    auto lmk_head2 = network->addConvolution(*ssh2->getOutput(0), 2 * 10, DimsHW{1, 1}, weightMap["LandmarkHead.1.conv1x1.weight"], weightMap["LandmarkHead.1.conv1x1.bias"]);
    auto lmk_head3 = network->addConvolution(*ssh3->getOutput(0), 2 * 10, DimsHW{1, 1}, weightMap["LandmarkHead.2.conv1x1.weight"], weightMap["LandmarkHead.2.conv1x1.bias"]);

    // ------------- Decode bbox, conf, landmark ---------------
    ITensor* inputTensors1[] = {bbox_head1->getOutput(0), cls_head1->getOutput(0), lmk_head1->getOutput(0)};
    auto cat1 = network->addConcatenation(inputTensors1, 3);
    ITensor* inputTensors2[] = {bbox_head2->getOutput(0), cls_head2->getOutput(0), lmk_head2->getOutput(0)};
    auto cat2 = network->addConcatenation(inputTensors2, 3);
    ITensor* inputTensors3[] = {bbox_head3->getOutput(0), cls_head3->getOutput(0), lmk_head3->getOutput(0)};
    auto cat3 = network->addConcatenation(inputTensors3, 3);
    auto decode = new DecodePlugin();
    ITensor* inputTensors[] = {cat1->getOutput(0), cat2->getOutput(0), cat3->getOutput(0)};
    auto decodelayer = network->addPlugin(inputTensors, 3, *decode);
    assert(decodelayer);
    decodelayer->setName("decode");

    decodelayer->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out, start building trt engine..." << std::endl;
    network->markOutput(*decodelayer->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
#ifdef USE_FP16
    builder->setFp16Mode(true);
#endif
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
        mem.second.values = NULL;
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

static inline cv::Rect get_rect_adapt_landmark2(cv::Mat& img, int input_w, int input_h, float bbox[4], float lmk[10]) {
    int l, r, t, b;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] / r_w;
        r = bbox[2] / r_w;
        t = (bbox[1] - (input_h - r_w * img.rows) / 2) / r_w;
        b = (bbox[3] - (input_h - r_w * img.rows) / 2) / r_w;
        for (int i = 0; i < 10; i += 2) {
            lmk[i] /= r_w;
            lmk[i + 1] = (lmk[i + 1] - (input_h - r_w * img.rows) / 2) / r_w;
        }
    } else {
        l = (bbox[0] - (input_w - r_h * img.cols) / 2) / r_h;
        r = (bbox[2] - (input_w - r_h * img.cols) / 2) / r_h;
        t = bbox[1] / r_h;
        b = bbox[3] / r_h;
        for (int i = 0; i < 10; i += 2) {
            lmk[i] = (lmk[i] - (input_w - r_h * img.cols) / 2) / r_h;
            lmk[i + 1] /= r_h;
        }
    }
    int w = r-l, h = b-t;
    l -= w * 1.1;
    r += w * 1.1;
    t -= h * 0.5 ;
    b += h * 1.2;
    if (l < 0) {
        l = 0;
    }
    if (t < 0) {
        t = 0;
    }
    if (r > img.cols) {
        r = img.cols;
    }
    if (b > img.rows) {
        b = img.rows;
    }
    return cv::Rect(l, t, r-l, b-t);
}

void detect(cv::Mat &img, retinaface5::detect::Response &response) {
    static float data[3 * INPUT_H * INPUT_W];
    cv::Mat pr_img = preprocess_img(img);
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = pr_img.at<cv::Vec3b>(i)[0] - 104.0;
        data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] - 117.0;
        data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[2] - 123.0;
    }

    // Run inference
    static float prob[OUTPUT_SIZE];
    std::vector<decodeplugin::Detection> res;
    res.clear();
    doInference(*context, data, prob, 1);
    nms(res, prob);
    std::cout << "detected before nms -> " << prob[0] << std::endl;
    std::cout << "after nms -> " << res.size() << std::endl;
    for (size_t j = 0; j < res.size(); j++) {
        if (res[j].class_confidence < 0.2) continue;
        cv::Rect r = get_rect_adapt_landmark(img, res[j].bbox, res[j].landmark);
        double esti_y = -2.7544 * r.height + 335.96;
	cv::putText(img, std::to_string(esti_y) + ", " + std::to_string(r.y)+ ", " +std::to_string(abs(esti_y - r.y)), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
        if (abs(esti_y - r.y) > 80) continue;
        double degree = (r.x + r.width / 2. - 320) * 0.195;
        response.degree.push_back(degree);
        int result = getGlass(img(get_rect_adapt_landmark2(img, INPUT_W, INPUT_H, res[j].bbox, res[j].landmark)));
        //res_pub.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", img(get_rect_adapt_landmark2(img, INPUT_W, INPUT_H, res[j].bbox, res[j].landmark))).toImageMsg());
        //response.hair.push_back(aero < 0.25);
        response.glass.push_back(result % 10 != 0);
        response.hair.push_back(result / 10 != 0);
        cv::rectangle(img, get_rect_adapt_landmark2(img, INPUT_W, INPUT_H, res[j].bbox, res[j].landmark), cv::Scalar(0x27, 0xC1, 0x36), 2);
         //cv::putText(img, std::to_string(r.height) + ", " + std::to_string(r.y), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
        //cv::putText(img, std::to_string(result), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
        for (int k = 0; k < 4; k += 2) {
            cv::circle(img, cv::Point(res[j].landmark[k], res[j].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
        }
    }
    res_pub.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg());
}

bool image_callback(retinaface5::detect::Request &req, retinaface5::detect::Response &res) {
    cv::Mat temp = cv_bridge::toCvCopy(req.img, sensor_msgs::image_encodings::BGR8) -> image;
    detect(temp, res);
    return true;
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file(std::string(PACK_PATH) + "/engine/retina_r50.engine", std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    std::cout << "read engine success\n";

    PluginFactory pf;
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, &pf);
    std::cout << "deserial engine success\n";
    //ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    std::cout << "get context success\n";
    assert(context != nullptr);
    init_glass_detect();

    ros::init(argc, argv, "retinaface");     //初始化ROS节点
    ros::NodeHandle n;
    ros::ServiceServer service = n.advertiseService("/detect", image_callback);
    res_pub = n.advertise<sensor_msgs::Image>("/image", 1);

    std::cout << "start identify\n";

    ros::spin();

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
