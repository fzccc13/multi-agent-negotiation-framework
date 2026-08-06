#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2; 


template<typename TYPE_X> class  KernelSigmoid {
    public:
        __aicore__ inline   KernelSigmoid () {}
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
            uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain,
            TPipe * pipeIn) {
            ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
            this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
            this->tileLength = block_size;
            this->ALIGN_NUM = ALIGN_NUM;
            this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
            auto startPointer = core_size * GetBlockIdx();
            auto bufferlength = this->blockLength;
            Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x + startPointer ,bufferlength);

            Gm_y.SetGlobalBuffer((__gm__ TYPE_X*)y + startPointer , bufferlength);
            
    
            this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
    
            pipe = pipeIn;
            pipe->InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
            pipe->InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
              if constexpr (std::is_same_v<TYPE_X, float>) {
                pipe->InitBuffer(tmp1,this->tileLength * sizeof(TYPE_X));
              }
              if constexpr (std::is_same_v<TYPE_X, half>) {
                pipe->InitBuffer(tmp1,this->tileLength * sizeof(float));
                pipe->InitBuffer(tmp2,this->tileLength * sizeof(float));
              }
              if constexpr (std::is_same_v<TYPE_X, int32_t>) {
                pipe->InitBuffer(tmp1,this->tileLength * sizeof(float));
                pipe->InitBuffer(tmp2,this->tileLength * sizeof(float));
              }
               if constexpr (std::is_same_v<TYPE_X, int16_t>) {
                pipe->InitBuffer(tmp1,this->tileLength * sizeof(float));
                pipe->InitBuffer(tmp2,this->tileLength * sizeof(float));
              }
            
        }
        __aicore__ inline void Process() { 

            int32_t loopCount = this->tileNum;
            for (int32_t i = 0; i < loopCount-1; i++) {
                CopyIn(i, this->tileLength);
                Compute(i, this->tileLength);
                CopyOut(i, this->tileLength);
            }
            uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
            CopyIn(loopCount - 1, length);
            Compute(loopCount - 1, length);
            CopyOut(loopCount - 1, length );
        }   
    
         
    private:
        __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
           LocalTensor<TYPE_X> x = Q_x.AllocTensor<TYPE_X>();
           
           DataCopy(x, Gm_x[progress * this->tileLength], length);           
           Q_x.EnQue(x);

        }
        __aicore__ inline void Compute(int32_t progress, uint32_t length) {
            LocalTensor<TYPE_X> xLocal = Q_x.DeQue<TYPE_X>();
          
            LocalTensor<TYPE_X> yLocal = Q_y.AllocTensor<TYPE_X>();

            if constexpr (std::is_same_v<TYPE_X, float>) {

                auto  one = tmp1.Get<TYPE_X>();

                Duplicate(one, static_cast<TYPE_X>(1.0), length);

                Muls(xLocal,xLocal,static_cast<TYPE_X>(-1.0),length);

                Exp(xLocal,xLocal,length);

                Adds(xLocal,xLocal,static_cast<TYPE_X>(1.0),length);

                Div(yLocal,one,xLocal,length);
            }
            else if constexpr (std::is_same_v<TYPE_X, half>) {
                auto  one = tmp1.Get<float>();

                Duplicate(one, static_cast<float>(1.0), length);

                LocalTensor<float> p1 = tmp2.Get<float>();
                Cast(p1,xLocal,AscendC::RoundMode::CAST_NONE, length);
               
                Muls(p1,p1,static_cast<float>(-1.0),length);

                Exp(p1,p1,length);

                Adds(p1,p1,static_cast<float>(1.0),length);

                Div(p1,one,p1,length);

                Cast(yLocal,p1,AscendC::RoundMode::CAST_NONE, length);


            }
            else if constexpr (std::is_same_v<TYPE_X, int32_t>) {

                auto  one = tmp1.Get<float>();

                Duplicate(one, static_cast<float>(1.0), length);

                LocalTensor<float> p1 = tmp2.Get<float>();
                Cast(p1,xLocal,AscendC::RoundMode::CAST_NONE, length);
               
                Muls(p1,p1,static_cast<float>(-1.0),length);

                Exp(p1,p1,length);

                Adds(p1,p1,static_cast<float>(1.0),length);

                Div(p1,one,p1,length);

                Cast(yLocal,p1,AscendC::RoundMode::CAST_FLOOR, length);

            }
            else if constexpr (std::is_same_v<TYPE_X, int16_t>) {

                auto  one = tmp1.Get<float>();

                Duplicate(one, static_cast<float>(1.0), length);

                LocalTensor<float> p1 = tmp2.Get<float>();
                Cast(p1,xLocal,AscendC::RoundMode::CAST_NONE, length);
               
                Muls(p1,p1,static_cast<float>(-1.0),length);

                Exp(p1,p1,length);

                Adds(p1,p1,static_cast<float>(1.0),length);

                Div(p1,one,p1,length);

                Cast(yLocal,p1,AscendC::RoundMode::CAST_FLOOR, length);


            }

            
            Q_x.FreeTensor(xLocal);  

            Q_y.EnQue<TYPE_X>(yLocal);
        }
        __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
            LocalTensor<TYPE_X> y = Q_y.DeQue<TYPE_X>();
            DataCopy(Gm_y[progress * this->tileLength], y, length);
            Q_y.FreeTensor(y);
        }
    private:
        TPipe * pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
        TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
        TBuf<QuePosition::VECCALC> tmp1,tmp2;
        GlobalTensor<TYPE_X> Gm_x,Gm_y;
        uint32_t blockLength;
        uint32_t tileNum;
        uint32_t tileLength;
        uint32_t ALIGN_NUM;
    };

extern "C" __global__ __aicore__ void sigmoid(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    TPipe pipe;
    KernelSigmoid<DTYPE_X> op;
    op.Init(x, y,
        tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain,&pipe);
    op.Process();
    // TODO: user kernel impl
}