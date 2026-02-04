#include <cstddef>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================

    // Adam update formulas:
    // m_t = beta1 * m_{t-1} + (1 - beta1) * grad
    // v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
    // m_hat = m_t / (1 - beta1^t)
    // v_hat = v_t / (1 - beta2^t)
    // param = param - learning_rate * m_hat / (sqrt(v_hat) + eps)

    for (int64_t idx = 0; idx < grad->NumElements(); ++idx) {
        // Get pointers to data
        float g = static_cast<const float *>(grad->DataPtr())[idx];
        float *p = &static_cast<float *>(param->DataPtr())[idx];
        float *m_ptr = &static_cast<float *>(m->DataPtr())[idx];
        float *v_ptr = &static_cast<float *>(v->DataPtr())[idx];

        // Update first moment (momentum)
        *m_ptr = beta1 * (*m_ptr) + (1.0f - beta1) * g;

        // Update second moment (RMSprop)
        *v_ptr = beta2 * (*v_ptr) + (1.0f - beta2) * g * g;

        // Bias correction
        float m_hat = (*m_ptr) / (1.0f - std::pow(beta1, t));
        float v_hat = (*v_ptr) / (1.0f - std::pow(beta2, t));

        // Update parameters
        *p = *p - learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
