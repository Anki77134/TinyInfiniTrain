#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

__global__ void AccumulateGradKernel(const float *grad_ptr, float rate, float *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += rate * grad_ptr[idx];
    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    const float *grad_ptr = static_cast<const float *>(gradient->DataPtr());
    float *tensor_ptr = static_cast<float *>(tensor->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, rate, tensor_ptr, num_elements);
}

__global__ void AdamKernel(float *grad, float *param, float *m, float *v, float lr, float beta1, float beta2, float eps,
                           float bias_correction1, float bias_correction2, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grad[idx];

        // Update first moment
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;

        // Update second moment
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

        // Bias correction
        float m_hat = m[idx] / bias_correction1;
        float v_hat = v[idx] / bias_correction2;

        // Update parameters
        param[idx] = param[idx] - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================

    // Calculate bias correction factors
    float bias_correction1 = 1.0f - std::pow(beta1, t);
    float bias_correction2 = 1.0f - std::pow(beta2, t);

    // Get number of elements
    int64_t num_elements = grad->NumElements();

    // Configure kernel launch parameters
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;

    // Launch kernel
    AdamKernel<<<grid_size, block_size>>>(static_cast<float *>(grad->DataPtr()),
                                          static_cast<float *>(param->DataPtr()),
                                          static_cast<float *>(m->DataPtr()), static_cast<float *>(v->DataPtr()),
                                          learning_rate, beta1, beta2, eps, bias_correction1, bias_correction2,
                                          num_elements);
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
