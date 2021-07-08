#include <iostream>
#include <chrono>
#include <map>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <algorithm>
#include <vector>
#include <cstring>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)
using namespace nvinfer1;

static Logger gLogger;

const int DEVICE = 2;
const std::string EngineName = "crnn.plan";
const int maxBatchSize = 8;
const std::string WeightName = "crnn.wts";
const std::string InputNames[] = {"INPUT__0"};
const std::string OutputNames[] = {"OUTPUT__0", "OUTPUT__1"};
const int NumClasses = 6625; // 6624 + 1(blank)
const int FeatureSize = 512;
const int InputWidth[] = {128, 192, 320, 512};
const int INDICES[] = {0, 1, 2, 3, 4, 5, 6, 7};

void print_dims(std::string name, Dims dim){
    if (dim.nbDims == 0){
        std::cout<<name<<" is scalar, has no dimension\n";
        return ;
    }
    std::cout<<name<<" shape -> [";
    for(int i=0; i<dim.nbDims; i++){
        std::cout<<dim.d[i]<<" ";
    }
    std::cout<<"]\n";
}

struct LstmIO
{
    nvinfer1::ITensor* data;
    nvinfer1::ITensor* hidden;
    nvinfer1::ITensor* cell;
};
struct LstmParams
{
    nvinfer1::ITensor* inputWeights;
    nvinfer1::ITensor* recurrentWeights;
    nvinfer1::ITensor* inputBias;
    nvinfer1::ITensor* recurrentBias;
    nvinfer1::ITensor* maxSequenceSize;
    int hiddenSize;
};

template<typename ScalarType>
nvinfer1::ILayer* addConst(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, 
                            Dims dims, std::vector<ScalarType> values, DataType dtype, 
                            std::string lname)
{
    size_t size = 1;
    for (int i=0; i<dims.nbDims; i++) size *= dims.d[i];
    assert(size == values.size() && "add const error!");

    Weights wt{dtype, nullptr, 0};
    wt.count = size;
    void* ptr = malloc(size*sizeof(ScalarType));
    wt.values = ptr;
    weightMap[lname] = wt;

    std::memcpy(ptr, values.data(), values.size() * sizeof(ScalarType));
    return network->addConstant(dims, wt);
}

nvinfer1::ILayer* addSingleIntValueConst(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, int value, std::string lname)
{
    Dims dims;
    dims.nbDims = 1;
    dims.d[0] = 1;

    Weights wt{DataType::kINT32, nullptr, 1};
    int* value_ptr = reinterpret_cast<int*> (malloc(sizeof(int)));
    *value_ptr = value;
    
    wt.values = value_ptr;
    weightMap[lname] = wt;

    return network->addConstant(dims, wt);
}

nvinfer1::ILayer* addAxisValue(INetworkDefinition *network, int axis, Dims dims={0})
{
    assert(0 <= axis && axis <= 7 && "addAxisValue error ! axis must between 0~7");
    Weights wt{DataType::kINT32, &INDICES[axis], 1};
    return network->addConstant(dims, wt);
}

nvinfer1::ITensor* getAxisLength(INetworkDefinition *network, ITensor* shape, int axis, Dims dims={0})
{
    ITensor* idx = addAxisValue(network, axis, dims)->getOutput(0);
    return network->addGather(*shape, *idx, 0)->getOutput(0);
}

nvinfer1::ITensor* unsqueezeTensor(INetworkDefinition *network, ITensor& input, int axis)
{
    ITensor* input_shape = network->addShape(input)->getOutput(0);
    Dims dims = input_shape->getDimensions();
    std::vector<ITensor*> shape_vec;

    for(int i=0; i<dims.nbDims; i++){
        if (i == axis){
            shape_vec.push_back(addAxisValue(network, i, Dims{1,1})->getOutput(0));
        } else {
            shape_vec.push_back(getAxisLength(network, input_shape, i, Dims{1,1}));
        }
    }
    ITensor* target_shape = network->addConcatenation(shape_vec.data(), shape_vec.size())->getOutput(0);
    auto unsqueeze = network->addShuffle(input);
    unsqueeze->setInput(1, *target_shape);
    return unsqueeze->getOutput(0);
}

nvinfer1::ITensor* constantOfShape(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, nvinfer1::ITensor* constant, nvinfer1::ITensor* shape, std::string lname)
{
    int rank = shape->getDimensions().d[0];

    std::vector<int> starts(rank);
    std::fill(starts.begin(), starts.end(), 0);

    nvinfer1::Dims strides{rank};
    std::fill(strides.d, strides.d + strides.nbDims, 0);

    // Slice will not work if constant does not have the same rank as start/size/strides.
    nvinfer1::Dims unsqueezeDims{rank};
    std::fill(unsqueezeDims.d, unsqueezeDims.d + unsqueezeDims.nbDims, 1);
    nvinfer1::IShuffleLayer* unsqueeze = network->addShuffle(*constant);
    unsqueeze->setReshapeDimensions(unsqueezeDims);
    unsqueeze->setZeroIsPlaceholder(false);
    constant = unsqueeze->getOutput(0);
    unsqueeze->setName((lname+".squeeze").c_str());

    // use zero stride to broadcast
    nvinfer1::ISliceLayer* broadcast = network->addSlice(*constant, nvinfer1::Dims{}, nvinfer1::Dims{}, strides);
    broadcast->setInput(1,
        *addConst(network, weightMap, nvinfer1::Dims{1, rank}, starts, DataType::kINT32, lname)->getOutput(0));

    broadcast->setInput(2, *shape);

    return broadcast->getOutput(0);
}

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

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

void splitLstmWeights(std::map<std::string, Weights>& weightMap, std::string lname) {
    // std::cout<<"split lstm weight : " <<lname<<"\n";
    int weight_size = weightMap[lname].count;
    for (int i = 0; i < 4; i++) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        wt.count = weight_size / 4;
        float *val = reinterpret_cast<float*>(malloc(sizeof(float) * wt.count));
        memcpy(val, (float*)weightMap[lname].values + wt.count * i, sizeof(float) * wt.count);
        wt.values = val;
        weightMap[lname + std::to_string(i)] = wt;
    }
}

nvinfer1::IScaleLayer* addNormlize(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input)
{
    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) ));
    for (int i = 0; i < 1; i++) {
        scval[i] = 2/255.0;
    }
    Weights scale{DataType::kFLOAT, scval, 1};

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) ));
    for (int i = 0; i < 1; i++) {
        shval[i] = -1;
    }
    Weights shift{DataType::kFLOAT, shval, 1};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float)));
    for (int i = 0; i < 1; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, 1};
    weightMap["norm.scale"] = scale;
    weightMap["norm.shift"] = shift;
    weightMap["norm.power"] = power;
    IScaleLayer* norm = network->addScale(input, ScaleMode::kUNIFORM, shift, scale, power);
    assert(norm && "add normlize preprocess");
    return norm;
}

nvinfer1::IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

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

nvinfer1::ILayer* addConv(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap, 
            ITensor& input, int nbOutChannels, Dims kernel_size, Dims stride, Dims padding,
            std::string lname, bool use_bn = false, bool use_relu=false)
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv = network->addConvolutionNd(input, nbOutChannels, kernel_size, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv && "add conv error!");
    conv->setStrideNd(stride);
    conv->setPaddingNd(padding);
    ILayer* last = conv;
    if (use_bn){
        last = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname+".bn", 1e-5);
        assert(last && "add batchNorm error!");
    }
    if (use_relu){
        last = network->addActivation(*last->getOutput(0), ActivationType::kRELU);
        assert(last && "add relu error!");
    }
    return last;
}

nvinfer1::ILayer* addShortCut(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, 
                            ITensor& input, int nbInChannels, int nbOutChannels, DimsHW stride, 
                            std::string lname, bool is_first=false)
{
    if ( (nbInChannels != nbOutChannels || stride.d[0] != 1) 
        && !is_first){
        // avg pool
        auto p = network->addPoolingNd(input, PoolingType::kAVERAGE, stride/*kernel_size*/);
        p->setStrideNd(stride);
        p->setPaddingNd(DimsHW{0, 0});
        p->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);

        // conv 
        auto y = addConv(network, weightMap, *p->getOutput(0), nbOutChannels, DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0}, lname+".conv", true,  false);
        return y;

    } else if (is_first){

        auto y = addConv(network, weightMap, input, nbOutChannels, DimsHW{1, 1}, stride, DimsHW{0, 0}, lname + ".conv", true, false);
        return y;

    } else {

        auto y = network->addIdentity(input);
        return y;

    }
}
nvinfer1::ILayer* addBasicBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, 
                        ITensor& input, int nbInChannels, int nbOutChannels, 
                        int stage_id, int depth, std::string lname)
{
    lname = lname + "." + std::to_string(stage_id) + "." + std::to_string(depth);
    bool is_first = false;
    if (stage_id == 0 && depth == 0) {
        is_first = true;
    }

    DimsHW stride;
    if (depth == 0 && stage_id != 0){
        stride = DimsHW{2, 1};
    } else {
        stride = DimsHW{1, 1};
    }

    // conv0, conv1
    ILayer* y = addConv(network, weightMap, input, nbOutChannels, DimsHW{3, 3}, stride, DimsHW{1, 1}, lname+".conv0", true, true);
    y = addConv(network, weightMap, *y->getOutput(0), nbOutChannels, DimsHW{3, 3}, DimsHW{1, 1}, DimsHW{1, 1}, lname+".conv1", true, false);

    // shortcut
    ILayer* shortcut = addShortCut(network, weightMap, input, nbInChannels, nbOutChannels, stride, lname+".shortcut", is_first);

    // conv1 + shortcut
    y = network->addElementWise(*y->getOutput(0), *shortcut->getOutput(0), ElementWiseOperation::kSUM);
    
    // relu
    y = network->addActivation(*y->getOutput(0), ActivationType::kRELU);
    return y;
}

nvinfer1::ILayer* addLoopLSTMCell(INetworkDefinition *network, std::map<std::string, Weights>& weightMap,
                                    const LstmIO& inputTensors, ITensor* seqLen, const LstmParams& params,
                                    LstmIO& outputTensors, std::string lname)
{
    ILoop* loop = network->addLoop();
    loop->addTripLimit(*seqLen, TripLimit::kCOUNT);
    print_dims(" triplimit ", seqLen->getDimensions());

    // [n, 512 or 256 /*featureSize*/]
    ITensor* input = loop->addIterator(*inputTensors.data, 0)->getOutput(0);
    print_dims(" iterator input ", input->getDimensions());

    IRecurrenceLayer* hidden = loop->addRecurrence(*inputTensors.hidden);
    IRecurrenceLayer* cell = loop->addRecurrence(*inputTensors.cell);

    // compute gates : gatesIFCO(t) = (X(t) * W^T + H(t-1) * R^T + (Wb + Rb)), 
    // pytorch's gates order is [input, forget, cell, output]
    
    // [n, 1024]
    nvinfer1::ITensor* xtWT = network->addMatrixMultiply(*input, MatrixOperation::kNONE,
                                         *params.inputWeights, MatrixOperation::kTRANSPOSE)
                                     ->getOutput(0);
    print_dims(" xtWT ", xtWT->getDimensions());

    nvinfer1::ITensor* ht1RT = network->addMatrixMultiply(*hidden->getOutput(0), MatrixOperation::kNONE,
                                          *params.recurrentWeights, MatrixOperation::kTRANSPOSE)
                                      ->getOutput(0);
    print_dims(" ht1RT ", ht1RT->getDimensions());

    // [1, 1024]
    ITensor* bias = network->addElementWise(*params.inputBias, *params.recurrentBias, ElementWiseOperation::kSUM)
              ->getOutput(0);
    // [n, 1024]
    ITensor* mm = network->addElementWise(*xtWT, *ht1RT, ElementWiseOperation::kSUM)->getOutput(0);

    // [n, 1024]
    ITensor* gatesIFCO = network->addElementWise(*mm, *bias, ElementWiseOperation::kSUM)->getOutput(0);
    print_dims(" gatesIFCO ", gatesIFCO->getDimensions());

    const auto isolateGate = [&](ITensor& gatesIFCO, int gateIndex) -> ITensor* {
        ISliceLayer* slice = network->addSlice(gatesIFCO, Dims2{0, gateIndex*256}, Dims2{1, 256}, Dims2{1, 1});
        ITensor* start = addConst(network, weightMap, Dims{1,2}, std::vector<int>{0, gateIndex*params.hiddenSize}, 
                            DataType::kINT32, lname +".isolateStart"+ std::to_string(gateIndex))
                        ->getOutput(0);
        ITensor* size = addConst(network, weightMap, Dims{1,2}, std::vector<int>{0, params.hiddenSize}, 
                            DataType::kINT32, lname + ".isolateSize" + std::to_string(gateIndex))
                        ->getOutput(0);
        // slice->setInput(1 , *start); need to fix in case of dynamic shape
        // slice->setInput(2 , *size); 
        return slice->getOutput(0);
    };

    // [n, hiddenSize]
    ITensor* iGate = network->addActivation(*isolateGate(*gatesIFCO, 0), ActivationType::kSIGMOID)->getOutput(0);
    ITensor* fGate = network->addActivation(*isolateGate(*gatesIFCO, 1), ActivationType::kSIGMOID)->getOutput(0);
    ITensor* cGate = network->addActivation(*isolateGate(*gatesIFCO, 2), ActivationType::kTANH)->getOutput(0);
    ITensor* oGate = network->addActivation(*isolateGate(*gatesIFCO, 3), ActivationType::kSIGMOID)->getOutput(0);

    // C(t) = f(t) . C(t-1) + i(t) . c(t)
    // [n, hiddenSize]
    ITensor* Ct = network->addElementWise(
        *network->addElementWise(*fGate, *cell->getOutput(0), ElementWiseOperation::kPROD)->getOutput(0),
        *network->addElementWise(*iGate, *cGate, ElementWiseOperation::kPROD)->getOutput(0),
        ElementWiseOperation::kSUM)
        ->getOutput(0);
    
    print_dims(" Ct ", Ct->getDimensions());

    // H(t) = o(t) . tanh(C(t))
    // [n, hiddenSize]
    ITensor* Ht = network->addElementWise(
        *oGate, 
        *network->addActivation(*Ct, ActivationType::kTANH)->getOutput(0), 
        ElementWiseOperation::kPROD)
        ->getOutput(0);

    print_dims(" Ht ", Ht->getDimensions());

    cell->setInput(1, *Ct);
    hidden->setInput(1, *Ht);

    // output [seqLen, n, hiddenSize]
    ILoopOutputLayer* outputLayer = loop->addLoopOutput(*Ht, nvinfer1::LoopOutput::kCONCATENATE, 0);
    outputLayer->setInput(1, *seqLen);

    // [n, hiddenSize]
    // ITensor* hiddenOut = loop->addLoopOutput(*hidden->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0);
    // ITensor* cellOut= loop->addLoopOutput(*cell->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0);

    // outputTensors = LstmIO{outputLayer->getOutput(0), hiddenOut, cellOut};
    outputTensors = LstmIO{outputLayer->getOutput(0), nullptr, nullptr};
    return outputLayer;
}

// only support undirectional forward mode;
//TODO : support bidirectional lstm
nvinfer1::ILayer* addLoopLSTMLayers(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, 
                                    ITensor* input,
                                    int numLayers, int hiddenSize, int featureSize, 
                                    std::string lname)
{
    ILayer* dataOut{nullptr};

    // inputShape {seqlen, n, 512 /*featureSize*/};
    ITensor* inputShape = network->addShape(*input)->getOutput(0);
    ITensor* seqLen = getAxisLength(network, inputShape, 0); // scalar, zero dims

    auto initialStateShape = [&]() -> ITensor*{
        // axis 1's length should be batchSize
        ITensor* numLayersTensor = addSingleIntValueConst(network, weightMap, numLayers, lname + ".numLayers")->getOutput(0);
        ITensor* batchSizeTensor = getAxisLength(network, inputShape, 1, Dims{1, 1});
        batchSizeTensor = addSingleIntValueConst(network, weightMap, 1, lname + ".batchSize")->getOutput(0);

        ITensor* hiddenSizeTensor = addSingleIntValueConst(network, weightMap, hiddenSize, lname + ".hiddenSize")->getOutput(0);

        std::array<ITensor*, 3> tensors{{numLayersTensor, batchSizeTensor, hiddenSizeTensor}};
        auto concat = network->addConcatenation(tensors.data(), tensors.size());
        return concat->getOutput(0);
    };

    ITensor* gateOutputShape = initialStateShape();

    // 0.0 constant
    ITensor* constant = addConst(network, weightMap, Dims{1,1}, std::vector<float>{0.0f}, DataType::kFLOAT, lname+".constantZero")
                        ->getOutput(0);

    ITensor* hidden0 = constantOfShape(network, weightMap, constant, gateOutputShape, lname+".hiddenZero");
    ITensor* cell0 = constantOfShape(network, weightMap, constant, gateOutputShape, lname+".cellZero");

    std::vector<ITensor*> hiddenOutputs, cellOutputs;
    LstmIO lstmOutput{input, nullptr, nullptr};

    auto getWeights = [&](std::string keyName, Dims dims) -> ITensor* {
        size_t size = 1;
        for(int i=0; i<dims.nbDims; i++) size *= dims.d[i];
        Weights weights = weightMap[keyName];
        // print_dims(keyName, dims);
        // std::cout<<"count : "<<weights.count<<"\n";
        assert(size == weights.count && "getWeights error! Shape doesn't match total count");
        return network->addConstant(dims, weights)->getOutput(0);
    };
    

    Dims2 dimsWL0{4 * hiddenSize, FeatureSize};
    Dims2 dimsR{4 * hiddenSize, hiddenSize};
    Dims2 dimsB{1, 4 * hiddenSize};

    for(int i=0; i<numLayers; i++){
        // []
        ITensor* index = addAxisValue(network, i)->getOutput(0);
        ITensor* initialHidden = network->addGather(*hidden0, *index, 0)->getOutput(0);
        ITensor* initialCellState = network->addGather(*cell0, *index, 0)->getOutput(0);;

        LstmIO lstmInput{lstmOutput.data, initialHidden, initialCellState};

        Dims dimsW = i == 0 ? dimsWL0 : dimsR;

        ITensor* weightIn = getWeights(lname + ".weight_ih_l" + std::to_string(i), dimsW);
        ITensor* weightRec = getWeights(lname + ".weight_hh_l" + std::to_string(i), dimsR);
        ITensor* biasIn = getWeights(lname + ".bias_ih_l" + std::to_string(i), dimsB);
        ITensor* biasRec = getWeights(lname + ".bias_hh_l" + std::to_string(i), dimsB);

        LstmParams params{weightIn, weightRec, biasIn, biasRec, seqLen, hiddenSize};

        dataOut = addLoopLSTMCell(network, weightMap, lstmInput, seqLen, params, lstmOutput, lname + ".layer" + std::to_string(i));


        // in crnn , we don't need h_n, c_n

        // push_back [1, n, hiddenSize]
        // hiddenOutputs.push_back(unsqueezeTensor(network, *lstmOutput.hidden, 0));
        // cellOutputs.push_back(unsqueezeTensor(network, *lstmOutput.cell, 0));
    }

    // auto addConcatenation = [&](std::vector<ITensor*>& tensors) -> ITensor* {
    //     auto concat = network->addConcatenation(tensors.data(), tensors.size());
    //     concat->setAxis(0);
    //     return concat->getOutput(0);
    // };

    // ITensor* hiddenOut = addConcatenation(hiddenOutputs);
    // ITensor* cellOut = addConcatenation(cellOutputs);

    return dataOut;
}


void constructNetwork(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network, nvinfer1::DataType dtype, std::map<std::string, Weights>& weightMap, int input_w)
{
    // for test or static shape
    int seqlen = input_w == -1 ? 64 : input_w / 4; 

    // extra_out for debug
    // nvinfer1::ITensor* extra_out;

    // nvinfer1::ITensor* input = network->addInput(InputNames[0], dtype, nvinfer1::Dims4{-1, 3, 32, -1});
    nvinfer1::ITensor* input = network->addInput(InputNames[0].c_str(), dtype, nvinfer1::Dims4{1, 3, 32, input_w});


    /************************************************** normalize ****************************************************************/

    ILayer* x = addNormlize(network, weightMap, *input);
    
    
    /************************************************** resnet-34 backone ********************************************************/
    // add conv: input, outChannels, kernel, stride, padding, layername, bn, relu
    // conv1 
    x = addConv(network, weightMap, *x->getOutput(0), 32, DimsHW{3, 3}, DimsHW{1, 1}, DimsHW{1, 1}, "backbone.conv1.0", true, true);
    x = addConv(network, weightMap, *x->getOutput(0), 32, DimsHW{3, 3}, DimsHW{1, 1}, DimsHW{1, 1}, "backbone.conv1.1", true, true);
    x = addConv(network, weightMap, *x->getOutput(0), 64, DimsHW{3, 3}, DimsHW{1, 1}, DimsHW{1, 1}, "backbone.conv1.2", true, true);

    // pool1 
    auto p = network->addPoolingNd(*x->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    p->setStrideNd(DimsHW{2, 2});
    p->setPaddingNd(DimsHW{1, 1});
    

    print_dims("after pool1 ", p->getOutput(0)->getDimensions());
    
    // backbone depth list
    x = p;
    int maxDepth[] = {3, 4, 6, 3};
    int numChannels[] = {64, 128, 256, 512};
    int pre_stage_id = 0;
    for (int stage_id = 0; stage_id < 4; stage_id ++){
        for(int depth = 0; depth < maxDepth[stage_id]; depth++){
            x = addBasicBlock(network, weightMap, 
                    *x->getOutput(0), numChannels[pre_stage_id], numChannels[stage_id], 
                    stage_id, depth, "backbone.stages"
            );
        }
        print_dims("after stage " + std::to_string(stage_id), x->getOutput(0)->getDimensions());
        pre_stage_id ++;
    }

    // maxpool 2
    p = network->addPoolingNd(*x->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    p->setStrideNd(DimsHW{2, 2});
    p->setPaddingNd(DimsHW{0, 0});
    
    print_dims("after maxpool out 2", p->getOutput(0)->getDimensions());

    /***************************************************** shape tensor for dynamic shape ****************************************************/

    // shapeTensor : {n, 512, 1, seqlen}
    ITensor* tensor_size = network->addShape(*p->getOutput(0))->getOutput(0);
    ITensor* idxs = addConst(network, weightMap, Dims{1, 3}, std::vector<int>{0, 1, 3}, DataType::kINT32, "idxs_1")->getOutput(0);
    ITensor* new_shape = network->addGather(*tensor_size, *idxs, 0)->getOutput(0);


    // squeeze + transpose, nchw -> nwc, h = 1, w = sequence_len, c = featureSize, [seqlen, n, 512]
    IShuffleLayer* sfl = network->addShuffle(*p->getOutput(0));
    // sfl->setInput(1, *new_shape);
    sfl->setReshapeDimensions(Dims3{1, 512, 64});
    sfl->setSecondTranspose(Permutation{2, 0, 1});
    sfl->setName("sfl");
    print_dims("lstm1 input", sfl->getOutput(0)->getDimensions());


    idxs = addConst(network, weightMap, Dims{1, 2}, std::vector<int>{0, 3}, DataType::kINT32, "idxs_2")->getOutput(0);
     // batch_seqlen_tensor, values: {n, seqlen}, dims: {nbDims=1, d={2}}, 
    ITensor* batch_seqlen_tensor = network->addGather(*tensor_size, *idxs, 0)->getOutput(0);


    /************************************************** lstm head *************************************************************/
    // lstm 1 out: [seqlen, n, 256 /*hiddenSize*/ ]
    ILayer* lstm1 = addLoopLSTMLayers(network, weightMap, sfl->getOutput(0), 2 /*numLayers*/, 256 /*hidden_size*/, FeatureSize /*512*/, "head.lstm1");
    print_dims("lstm1 out", lstm1->getOutput(0)->getDimensions());
    
    // lstm 2 , need reverse input and output
    ITensor* slice_size = network->addShape(*sfl->getOutput(0))->getOutput(0);
    auto slice = network->addSlice(*sfl->getOutput(0), Dims3{63, 0, 0}, Dims3{seqlen, 1, FeatureSize}, Dims3{-1, 1, 1});
    assert(slice && "add slice 1 error");
    slice->setMode(SliceMode::kWRAP);
    // slice->setInput(2, *slice_size);
    print_dims("lstm2 input ", slice->getOutput(0)->getDimensions());
    
    // out : [seqlen, n, 256 /*hiddenSize*/ ]
    ILayer* lstm2 = addLoopLSTMLayers(network, weightMap, slice->getOutput(0), 2 /*numLayers*/, 256 /*hidden_size*/, FeatureSize /*512*/, "head.lstm2");
    print_dims("lstm2 output ", lstm2->getOutput(0)->getDimensions());

    slice = network->addSlice(*lstm2->getOutput(0), Dims3{63, 0, 0}, Dims3{seqlen, 1, 256 /*hiddenSize*/}, Dims3{-1, 1, 1});
    assert(slice && "add slice 2 error");
    slice->setMode(SliceMode::kWRAP);
    slice_size = network->addShape(*lstm2->getOutput(0))->getOutput(0);
    // slice->setInput(2, *slice_size);

    // concat [seqlen, n, 256] [seqlen, n, 256] => [seqlen, n, 512]
    std::array<ITensor*, 2> tensors{lstm1->getOutput(0), slice->getOutput(0)};
    auto concat = network->addConcatenation(tensors.data(), tensors.size());
    concat->setAxis(2);
    print_dims("after concat", concat->getOutput(0)->getDimensions());

    // reshape, [seqlen, n, 512] -> [n*seqlen, 1, 1, 512]
    auto sfl_2 = network->addShuffle(*concat->getOutput(0));
    sfl_2->setFirstTranspose(Permutation{1, 0, 2});
    sfl_2->setReshapeDimensions(Dims4{-1, 1, 1, FeatureSize});
    sfl_2->setName("sfl_2");
    print_dims("after shuffle 2", sfl_2->getOutput(0)->getDimensions());

    // fc, shape: [n*seqlen, NumClasses, 1, 1]
    x = network->addFullyConnected(*sfl_2->getOutput(0), NumClasses, weightMap["head.fc.weight"], weightMap["head.fc.bias"]);
    print_dims("after fc", x->getOutput(0)->getDimensions());


    ///****/ dynamic reshape: [n*seqlen, NumClasses, 1, 1] -> [n, seqlen, NumClasses]
    auto sfl_3 = network->addShuffle(*x->getOutput(0));
    sfl_3->setName("sfl_3");
    sfl_3->setReshapeDimensions(Dims3{-1, seqlen, NumClasses}); //static reshape for debug

    ITensor* const_num_classes_ts = addSingleIntValueConst(network, weightMap, NumClasses, "const_num_classes_ts")->getOutput(0);
    std::array<ITensor*, 2> shape_tensors = {{batch_seqlen_tensor, const_num_classes_ts}};
    ITensor* fc_shape_tensor = network->addConcatenation(shape_tensors.data(), shape_tensors.size())->getOutput(0);
    // sfl_3->setInput(1, *fc_shape_tensor);
    print_dims("after shuffle 3", sfl_3->getOutput(0)->getDimensions());

    // softmax : shape [n, seqlen, NumClasses]
    auto softmax = network->addSoftMax(*sfl_3->getOutput(0));
    softmax->setAxes(1<<2);
    print_dims("after softmax ", softmax->getOutput(0)->getDimensions());

    // top1 layer
    ILayer* top_1 = network->addTopK(*softmax->getOutput(0), TopKOperation::kMAX, 1, 1<<2);

    // OUTPUT__0 : idx, shape : [n, seqlen, 1] -> [n, seqlen]
    auto sf = network->addShuffle(*top_1->getOutput(1));
    sf->setName("sf");
    sf->setReshapeDimensions(Dims2{-1, seqlen}); // static reshape for debug

    // dynamic reshape
    // sf->setInput(1, *batch_seqlen_tensor);
    sf->getOutput(0)->setName(OutputNames[0].c_str());
    network->markOutput(*sf->getOutput(0));
    print_dims("idx output ", sf->getOutput(0)->getDimensions());


    // OUTPUT__1 : score, shape : [n, seqlen, 1] -> [n, seqlen]
    sf = network->addShuffle(*top_1->getOutput(0));
    sf->setReshapeDimensions(Dims2{-1, seqlen}); // static reshape for debug

    // sf->setInput(1, *batch_seqlen_tensor);
    sf->getOutput(0)->setName(OutputNames[1].c_str());
    network->markOutput(*sf->getOutput(0));
    print_dims("score output ", sf->getOutput(0)->getDimensions());

    // extra_out : for debug
    // extra_out = network->addIdentity(*extra_out)->getOutput(0);
    // extra_out->setName("extra_out");
    // network->markOutput(*extra_out);

}

void APIToModel(unsigned int maxBatchSize) {
    // Create builder
    nvinfer1::IBuilder* builder = createInferBuilder(gLogger);

    // weightMap
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(WeightName);
    std::cout<<"step 1: load weight done!\n";
    
    // network
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    constructNetwork(builder, network, nvinfer1::DataType::kFLOAT, weightMap, 256);
    std::cout<<"step 2 : constuct network done!\n";

    // config
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(12LL * (1<<30));

    // optimize profile
    const int profile_num = 1;
    // nvinfer1::Dims4 mi[] = {nvinfer1::Dims4{1, 3, 32, 32}, };
    // nvinfer1::Dims4 opt[] = {nvinfer1::Dims4{4, 3, 32, 256}, };
    // nvinfer1::Dims4 mx[] = {nvinfer1::Dims4{maxBatchSize, 3, 32, 512}, };
    nvinfer1::Dims4 mi[] = {nvinfer1::Dims4{1, 3, 32, 256}, };
    nvinfer1::Dims4 opt[] = {nvinfer1::Dims4{1, 3, 32, 256}, };
    nvinfer1::Dims4 mx[] = {nvinfer1::Dims4{1, 3, 32, 256}, };
    for (int i=0; i<profile_num; i++){
        auto profile = builder->createOptimizationProfile();
        profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, mi[i]);
        profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, opt[i]);
        profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, mx[i]);
        config->addOptimizationProfile(profile);
    }

    // build engine
    std::cout << "step 3 : building engine, please wait for a while...\n";
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine != nullptr && "build engine error ! ");
    std::cout << "Build engine successfully! \n";

    // Serialize the engine
    nvinfer1::IHostMemory* modelStream = engine->serialize();
    std::ofstream p(EngineName, std::ios::binary);
    assert(p && "write engine file failed!");
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    p.close();

    // Close everything down
    network->destroy();
    engine->destroy();
    builder->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }
}

void doInference(ICudaEngine* engine, IExecutionContext* context, cudaStream_t stream, void **buffers,
            float* input, int* output_0, float* output_1, 
            size_t input_size, size_t output_size,
            size_t extra_out_size, float* extra_out = nullptr)
{
    
    CHECK(cudaMemcpyAsync(buffers[0], input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream));
    cudaStreamSynchronize(stream);
    bool ok = false;
    
    for (int i=0; i<5; i++){
        ok = context->enqueueV2(buffers, stream, nullptr);
    }

    cudaStreamSynchronize(stream);

    auto start = std::chrono::system_clock::now();
    for (int i=0; i<30; i++){
        ok = context->enqueueV2(buffers, stream, nullptr);
    }
    cudaStreamSynchronize(stream);

    auto end = std::chrono::system_clock::now();

    assert(ok && "enqueuev2 inference error !");

    std::cout <<"inference avg time cost : "<< (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 1.0) / 30 << "ms\n";

    // tensorrt network output bingdings' order are not deterministic, need to double check bindingName 
    for(int i=1; i<engine->getNbBindings(); i++){
        if(std::string(engine->getBindingName(i)) == "OUTPUT__0"){
            CHECK(cudaMemcpyAsync(output_0, buffers[i], output_size * sizeof(int), cudaMemcpyDeviceToHost, stream));
        } else if(std::string(engine->getBindingName(i)) == "OUTPUT__1"){
            CHECK(cudaMemcpyAsync(output_1, buffers[i], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        } else {
            // extra_out for debug;
            CHECK(cudaMemcpyAsync(extra_out, buffers[i], extra_out_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
    }
    
    cudaStreamSynchronize(stream);
}

int main(int argc, char** argv)
{
    cudaSetDevice(DEVICE);
    std::vector<char> trtModelStream;

    if (argc == 2 && std::string(argv[1]) == "-s") {
        APIToModel(maxBatchSize);
        return 0;
    } else if (argc == 2 && std::string(argv[1]) == "-d") {
        std::ifstream file(EngineName, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size_t size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./crnn -s  // serialize model to plan file" << std::endl;
        std::cerr << "./crnn -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    const Dims4 input_shape = Dims4{1, 3, 32, 256};

    const size_t input_size = 1UL * input_shape.d[0] * input_shape.d[1] * input_shape.d[2] * input_shape.d[3];
    const size_t output_size = 1UL * input_shape.d[0] * input_shape.d[3] / 4;
    const size_t extra_out_size = 256*32*3;
    
    float data[input_size];
    float score[output_size];
    int idxs[output_size];
    float extra_out[extra_out_size];

    void* buffers[4];
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream.data(), trtModelStream.size());
    IExecutionContext* context = engine->createExecutionContext();

    for (int i=0; i<engine->getNbBindings(); i++){
        std::cout<<" "<<engine->getBindingName(i)<<" "<< (int)(engine->getBindingDataType(i))<<"\n";
    }

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[0], input_size * sizeof(float)));
    CHECK(cudaMalloc(&buffers[1], output_size * sizeof(int)));
    CHECK(cudaMalloc(&buffers[2], output_size * sizeof(float)));
    if(engine->getNbBindings() > 3){
        CHECK(cudaMalloc(&buffers[3], extra_out_size * sizeof(float)));
    }

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // 
    cv::Mat img = cv::imread("../id.jpg");
    if (img.empty()) {
        std::cerr << "id.jpg not found !!!" << std::endl;
        return 0;
    }

    cv::cvtColor(img, img, CV_BGR2RGB);
    for(int i=0; i<3; i++){
        for(int j=0; j<input_shape.d[2]; j++){
            for(int k=0; k<input_shape.d[3]; k++){
                data[i* (input_shape.d[2] * input_shape.d[3]) + j * (input_shape.d[3]) + k] = static_cast<float> (img.at<uchar>(j, k, i));
            }
        }
    }

    std::cout<<"input_data : ";
    for (int i=0; i<20; i++){
        std::cout<<data[i]<<" ";
    }
    std::cout<<"\n";
    
    context->setBindingDimensions(0, input_shape);
    
    doInference(engine, context, stream, buffers,
                data, idxs, score, 
                input_size, output_size,
                extra_out_size, nullptr);

    print_dims("idxs : ", context->getBindingDimensions(1));
    for(int i=0; i<output_size; i++){
        std::cout<<idxs[i]<<" ";
    }
    std::cout<<"\n\n";

    print_dims("score : ", context->getBindingDimensions(2));
    for(int i=0; i<output_size; i++){
        std::cout<<score[i]<<" ";
    }
    std::cout<<"\n\n";

    if(engine->getNbBindings() > 3){
        print_dims("extra_out : ", context->getBindingDimensions(3));
        std::cout<<"extra_out : ";
        for(int i=0; i<std::min(extra_out_size, 64UL); i++){
            std::cout<<extra_out[i]<<" ";
        }
        std::cout<<"\n\n";
    }

    for(int i=0; i<engine->getNbBindings(); i++) {
        CHECK(cudaFree(buffers[i]));
    }

    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}