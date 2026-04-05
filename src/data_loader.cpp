#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <random>

namespace rnet {

class DataLoader {
public:
    bool load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "Cannot open data file: %s\n", path.c_str());
            return false;
        }
        fseek(f, 0, SEEK_END);
        size_t size = ftell(f);
        fseek(f, 0, SEEK_SET);

        data_.resize(size);
        fread(data_.data(), 1, size, f);
        fclose(f);

        printf("Loaded %zu bytes from %s\n", size, path.c_str());
        return true;
    }

    // Generate a random batch of (input, target) pairs
    // For byte-level LM: input[t] = data[pos+t], target[t] = data[pos+t+1]
    void get_batch(int* h_input, int* h_target,
                   int batch_size, int seq_len, std::mt19937& rng) {
        if (data_.empty()) {
            // Generate synthetic data for testing
            for (int b = 0; b < batch_size; b++) {
                for (int t = 0; t < seq_len; t++) {
                    h_input[b * seq_len + t] = rng() % 256;
                    h_target[b * seq_len + t] = rng() % 256;
                }
            }
            return;
        }

        std::uniform_int_distribution<size_t> dist(0, data_.size() - seq_len - 2);
        for (int b = 0; b < batch_size; b++) {
            size_t pos = dist(rng);
            for (int t = 0; t < seq_len; t++) {
                h_input[b * seq_len + t]  = (int)(uint8_t)data_[pos + t];
                h_target[b * seq_len + t] = (int)(uint8_t)data_[pos + t + 1];
            }
        }
    }

    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

private:
    std::vector<char> data_;
};

} // namespace rnet
