#include "retinaface5/plugin_factory.h"
#include "NvInferPlugin.h"
#include "retinaface5/decode.h"
#include "retinaface5/common.h"

using namespace nvinfer1;
using nvinfer1::PluginFactory;

IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength) {
    IPlugin *plugin = nullptr;
    if (strstr(layerName, "leaky") != NULL) {
        plugin = plugin::createPReLUPlugin(serialData, serialLength);
    } else if (strstr(layerName, "decode") != NULL) {
        plugin = new DecodePlugin(serialData, serialLength);
    }
    return plugin;
}
