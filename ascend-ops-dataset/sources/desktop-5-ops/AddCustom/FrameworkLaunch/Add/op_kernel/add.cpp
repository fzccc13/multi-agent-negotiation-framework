#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2; 
template<typename TYPE_X> class KernelAdd{
    public:
        __aicore__ inline KernelAdd() {}
        __aicore__ inline void Init(GM_ADDR x1,GM_ADDR x2, GM_ADDR y,
            uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain,
            TPipe * pipeIn) {
            ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
            this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
            this->tileLength = block_size;
            this->ALIGN_NUM = ALIGN_NUM;
            this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
            auto startPointer = core_size * GetBlockIdx();
            auto bufferlength = this->blockLength;
            Gm_x1.SetGlobalBuffer((__gm__ TYPE_X*)x1 + startPointer ,bufferlength);
            Gm_x2.SetGlobalBuffer((__gm__ TYPE_X*)x2 + startPointer ,bufferlength);
            Gm_y.SetGlobalBuffer((__gm__ TYPE_X*)y + startPointer , bufferlength);
            this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
            pipe = pipeIn;
            pipe->InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
            pipe->InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
            pipe->InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
            if constexpr (std::is_same_v<TYPE_X, half> || std::is_same_v<TYPE_X, int16_t>) {
                pipeIn->InitBuffer(tmp1, this->tileLength * sizeof(float));
                pipeIn->InitBuffer(tmp2, this->tileLength * sizeof(float));
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
           LocalTensor<TYPE_X> x1 = Q_x1.AllocTensor<TYPE_X>();
           LocalTensor<TYPE_X> x2 = Q_x2.AllocTensor<TYPE_X>();
           DataCopy(x1, Gm_x1[progress * this->tileLength], length);
           DataCopy(x2, Gm_x2[progress * this->tileLength], length);               
           Q_x1.EnQue(x1);
           Q_x2.EnQue(x2);

        }
        __aicore__ inline void Compute(int32_t progress, uint32_t length) {
            LocalTensor<TYPE_X> x1Local = Q_x1.DeQue<TYPE_X>();
            LocalTensor<TYPE_X> x2Local = Q_x2.DeQue<TYPE_X>();
            LocalTensor<TYPE_X> yLocal = Q_y.AllocTensor<TYPE_X>();
            if constexpr (std::is_same_v<TYPE_X,half> || std::is_same_v<TYPE_X,int16_t>){
                LocalTensor<float> p1 = tmp1.Get<float>();
                LocalTensor<float> p2 = tmp2.Get<float>();
                Cast(p1,x1Local,AscendC::RoundMode::CAST_NONE, length);
                Cast(p2,x2Local,AscendC::RoundMode::CAST_NONE, length);
                Add(p1,p1,p2,length);
                Cast(yLocal,p1,AscendC::RoundMode::CAST_NONE, length);
            }
            else{
                Add(yLocal,x1Local,x2Local,length);

            }        
            Q_x1.FreeTensor(x1Local);  
            Q_x2.FreeTensor(x2Local);  
            Q_y.EnQue<TYPE_X>(yLocal);
        }
        __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
            LocalTensor<TYPE_X> y = Q_y.DeQue<TYPE_X>();
            DataCopy(Gm_y[progress * this->tileLength], y, length);
            Q_y.FreeTensor(y);
        }
    private:
        TPipe * pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1,Q_x2;
        TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
        TBuf<QuePosition::VECCALC> tmp1,tmp2;
        GlobalTensor<TYPE_X> Gm_x1,Gm_x2,Gm_y;
        uint32_t blockLength;
        uint32_t tileNum;
        uint32_t tileLength;
        uint32_t ALIGN_NUM;
    };
extern "C" __global__ __aicore__ void add(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelAdd<DTYPE_X1> op;
    op.Init(x1,x2, y,
        tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain,&pipe);
    op.Process();
}