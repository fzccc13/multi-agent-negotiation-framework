
#include "asinh_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>


namespace optiling {
    const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  AsinhTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t NUM = 1;
  uint32_t sizeofdatatype;
  uint32_t totalLengthAligned;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    //auto socVersion = ascendcPlatform.GetSocVersion();
  uint64_t ub_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  auto aivNum = ascendcPlatform.GetCoreNum();

  uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();

     //获取输入shape信息
  uint32_t inputNum = context->GetInputShape(1)->GetStorageShape().GetShapeSize(); //输入数量
  uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(1)->GetDataType()); //输入类型
  uint32_t inputLength = inputBytes * inputNum; //输入长度


  auto dt = context->GetInputTensor(0)->GetDataType();
 
  if(dt == ge::DT_FLOAT16){

    sizeofdatatype = 2;
    NUM = 11;
    }
  else{ 
    sizeofdatatype = 4;
    NUM = 5;
    }

  uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
  uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
  tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;

  uint32_t block_size = tiling_size * ALIGN_NUM;


  uint32_t core_size = (totalLength) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);
  uint32_t core_remain = totalLength - core_size;

  tiling.set_totalLength(totalLength);
  tiling.set_ALIGN_NUM(ALIGN_NUM);
  tiling.set_tiling_size(tiling_size);
  tiling.set_block_size(block_size);
  tiling.set_core_size(core_size);
  tiling.set_core_remain(core_remain);
  context->SetBlockDim(1);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Asinh : public OpDef {
public:
    explicit Asinh(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Asinh);
}
