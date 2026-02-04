#include "example/common/tokenizer.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "glog/logging.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

using Version = Tokenizer::Version;

const std::unordered_map<uint32_t, uint32_t> kEotMap = {
    {20240328, kGpt2Eot},   // GPT-2
    {20240801, kLLaMA3Eot}, // LLaMA-3
};

const std::unordered_map<uint32_t, std::vector<uint32_t>> kPromptMap = {
    // e.g. "The meaning of life is"
    // ref: https://tiktokenizer.vercel.app/
    {20240328, std::vector<uint32_t>{464, 3616, 286, 1204, 318}}, // GPT-2
    {20240801, std::vector<uint32_t>{791, 7438, 315, 2324, 374}}, // LLaMA-3
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

unsigned int RandomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
    return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== 作业 =====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */

    // 1. Open file
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open tokenizer file: " << filepath;

    // 2. Read header (1024 bytes)
    auto header = ReadSeveralBytesFromIfstream(1024, &ifs);

    // 3. Parse header fields
    int magic = BytesToType<int>(header, 0);
    int version = BytesToType<int>(header, 4);
    int vocab_size = BytesToType<int>(header, 8);

    magic_number_ = magic;
    vocab_size_ = vocab_size;

    // 4. Get EOT token based on version
    CHECK(kEotMap.count(magic)) << "Unknown magic number: " << magic;
    eot_token_ = kEotMap.at(magic);

    // 5. Read vocabulary table
    token_table_.resize(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        // Read string length (4 bytes)
        auto len_bytes = ReadSeveralBytesFromIfstream(4, &ifs);
        int str_len = BytesToType<int>(len_bytes, 0);

        // Read string data
        auto str_bytes = ReadSeveralBytesFromIfstream(str_len, &ifs);
        token_table_[i] = std::string(str_bytes.begin(), str_bytes.end());
    }

    ifs.close();
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */

    CHECK_LT(token_id, vocab_size_) << "Token ID out of range: " << token_id;
    return token_table_[token_id];
}

void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (int i = 0; i < batch_size * sequence_length; ++i) { x_buff[i] = eot_token_; }

    // Give some contexts: "The meaning of life is "
    auto prompt = kPromptMap.at(magic_number_);
    auto prompt_len = prompt.size();
    for (int i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";

    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    uint64_t rng_state = kRngState;  // Initialize from global constant
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */

        // 1. Forward pass to get logits for current position
        auto logits = model.Forward({x})[0]->Slice(-1, t, t + 1);

        // 2. Apply softmax to get probabilities
        auto probs = infini_train::nn::function::Softmax(logits, -1);

        // Move probs to CPU for sampling
        auto probs_cpu = probs->To(Device(DeviceType::kCPU, 0));

        // 3. Sample next token from probability distribution
        float *probs_data = static_cast<float *>(probs_cpu.DataPtr());
        float coin = RandomF32(rng_state);  // Use local rng_state
        uint32_t next_token = SampleMult(probs_data, vocab_size_, coin);

        // 4. Decode and print the token
        std::cout << Decode(next_token);
        std::cout.flush();

        // 5. Update input for next iteration (if not at end)
        if (t + 1 < sequence_length) {
            x_buff[t + 1] = next_token;
        }
    }
    std::cout << std::endl;
}
} // namespace infini_train
