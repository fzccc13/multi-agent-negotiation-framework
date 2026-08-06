#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2; 


template<typename TYPE_X> class KernelAsinh {
    public:
        __aicore__ inline  KernelAsinh() {}
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

            pipe->InitBuffer(tmpBuffer,this->tileLength * sizeof(DTYPE_X));

            if constexpr (std::is_same_v<TYPE_X, half>)
            {
                
                pipe->InitBuffer(tmp1, this->tileLength * sizeof(float));
                pipe->InitBuffer(tmp2, this->tileLength * sizeof(float));
                pipe->InitBuffer(tmp3, this->tileLength * sizeof(float));
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


            

           LocalTensor<DTYPE_X> tmp = tmpBuffer.Get<DTYPE_X>();

        
        if constexpr (std::is_same_v<DTYPE_X, half>){


            auto p1 = tmp1.Get<float>();
            auto p2 = tmp2.Get<float>();
            auto p3 = tmp3.Get<float>();

            Cast(p1, xLocal, RoundMode::CAST_NONE, length);

            Cast(p2, xLocal, RoundMode::CAST_NONE, length);

            Cast(p3, xLocal, RoundMode::CAST_NONE, length);
            
            Maxs(p2,p1,static_cast<float>(0),length);

            Mul(p3,p2,p2,length);
            Adds(p3,p3,static_cast<float>(1.0),length);

            Sqrt(p3,p3,length);

            Add(p2,p2,p3,length);

            Ln(p2,p2,length);



            Mins(p3,p1,static_cast<float>(0),length);

            Mul(p1,p3,p3,length);

            Adds(p1,p1,static_cast<float>(1.0),length);

            Sqrt(p1,p1,length);

            Sub(p1,p1,p3,length);

            Ln(p1,p1,length);

            Muls(p1,p1,static_cast<float>(-1.0),length);

            Add(p1,p2,p1,length);

            Cast(yLocal, p1, RoundMode::CAST_NONE, length);   





        }

        else{


        

        Maxs(tmp,xLocal,static_cast<DTYPE_X>(0),length);

        Mul(yLocal,tmp,tmp,length);
        Adds(yLocal,yLocal,static_cast<DTYPE_X>(1.0),length);

        Sqrt(yLocal,yLocal,length);

        Add(tmp,yLocal,tmp,length);

        Ln(tmp,tmp,length);


        Mins(xLocal,xLocal,static_cast<DTYPE_X>(0),length);

        Mul(yLocal,xLocal,xLocal,length);

        Adds(yLocal,yLocal,static_cast<DTYPE_X>(1.0),length);

        Sqrt(yLocal,yLocal,length);

        Sub(yLocal,yLocal,xLocal,length);

        Ln(yLocal,yLocal,length);

        Muls(yLocal,yLocal,static_cast<DTYPE_X>(-1.0),length);

        Add(yLocal,yLocal,tmp,length);

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

        TBuf<QuePosition::VECCALC> tmpBuffer,tmp1,tmp2,tmp3;
        TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
        TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
        GlobalTensor<TYPE_X> Gm_x,Gm_y;
        uint32_t blockLength;
        uint32_t tileNum;
        uint32_t tileLength;
        uint32_t ALIGN_NUM;
    };

extern "C" __global__ __aicore__ void asinh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    TPipe pipe;
    KernelAsinh<DTYPE_X> op;
    op.Init(x, y,
        tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain,&pipe);
    op.Process();
}