#include "fmod_tiling.h" //导入具体算子的切分数据头文件
#include "register/op_def_registry.h"//算子定义及算子注册的头文件
#include "tiling/platform/platform_ascendc.h"// 芯片内存相关头文件
#include <algorithm>// 算法相关头文件
namespace optiling {//算子切分核心代码块
    const uint32_t BLOCK_SIZE = 32; // 数据搬运及大部分昇腾API使用需要满足32字节对齐
static ge::graphStatus TilingFunc(gert::TilingContext* context) //算子数据切分核心逻辑
{
  FmodTilingData tiling; //定义 tiling 对象
  const gert::StorageShape* x1_shape = context->GetInputShape(0); // 获取第一个输入数据的形状
  int32_t NUM = 1; // 临时缓冲区数量（根据算子的实际功能进行调整）
  uint32_t sizeofdatatype; // 数据类型的字节大小（int8->1字节，float16->2字节，float32->4字节）
  uint32_t totalLengthAligned; //对齐后的输入数据长度
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    //auto socVersion = ascendcPlatform.GetSocVersion();//获取芯片平台的相关信息（内存大小、AI核的数量等等）
  uint64_t ub_size; //定义芯片内存的变量，表示芯片的实际内存大小，不同AI处理器的ub_size存在差异；
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size); //该方法将实际芯片的内存大小赋值给ub_size;
  auto aivNum = ascendcPlatform.GetCoreNum();// 获取实际芯片的核数,本操作手册使用的是310B4环境，默认为单核环境，即该数值为1；
  uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
     //获取输入shape信息
  uint32_t inputNum = context->GetInputShape(1)->GetStorageShape().GetShapeSize(); //输入数量
  uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(1)->GetDataType()); //输入类型
  uint32_t inputLength = inputBytes * inputNum; //输入长度
  auto dt = context->GetInputTensor(0)->GetDataType();//获取输入数据的实际数据类型
if(dt == ge::DT_FLOAT16){ //数据类型字节数均为2
sizeofdatatype = 2;// 数据类型字节数均为2
    NUM = 12; //设置数量需要在kernel侧确定（是否设置 double buffer,队列数量是多少，临时缓冲区的数量是多少）
    }
  else if  (dt == ge::DT_FLOAT){
    sizeofdatatype = 4;
    NUM = 7;
    }
  uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype; //计算对齐32B的实际输入数据需要多少个元素个数
uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM ; //只占用一半缓存的每个Tile的块数量
tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8; //256B对齐
uint32_t block_size = tiling_size * ALIGN_NUM; //每个切分Tile块的元素数量，即每次处理多少数据量
uint32_t core_size = (totalLength) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8); //实际输入的总数据量进行256B对齐
uint32_t core_remain = totalLength - core_size; //尾部数据需要处理的数量
tiling.set_ALIGN_NUM(ALIGN_NUM); // 参数传递至Kernel侧
tiling.set_block_size(block_size); // 参数传递至Kernel侧
tiling.set_core_size(core_size); // 参数传递至Kernel侧
tiling.set_core_remain(core_remain); // 参数传递至Kernel侧
context->SetBlockDim(1);//设置核的数量
tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
context->GetRawTilingData()->SetDataSize(tiling.GetDataSize()); return ge::GRAPH_SUCCESS;
}
}
namespace ge {  //推导输入数据形状
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{   //推导输入数据类型
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}
namespace ops { //算子定义及注册
class Fmod : public OpDef {
public:
    explicit Fmod(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(Fmod);
}
