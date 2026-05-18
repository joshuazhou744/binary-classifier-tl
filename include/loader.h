#pragma once

#include <tl/tensor.h>

#include <string>
#include <vector>
#include <utility>

// mel-spectrogram config
struct MelConfig {
  int sample_rate = 16000;
  int n_fft = 1024;
  int hop_length = 512;
  int n_mels = 64;
  float clip_duration = 15.0f; // pad/truncate all samples to this duration
  float db_floor = -80.0f; // decibel floor
};

// compute log-mel spectrogram
// returns: (1, n_mels, n_frames)
tl::Tensor compute_log_mel(const std::vector<float>& samples, const MelConfig& config);

// read labels.csv
// input format: file_id,label
std::vector<std::pair<std::string, int>> read_labels(const std::string& csv_path);

// load all training samples and labels from directory of WAV files and labels CSV
void load_train(
    const std::string& wav_dir,
    const std::string& labels_csv,
    const MelConfig& config,
    int max_count,
    tl::Tensor& x,
    std::vector<int>& labels
);

// load all holdout samples from directory of WAV files
void load_holdout(
    const std::string& wav_dir,
    const MelConfig& config,
    int max_count,
    tl::Tensor& x,
    std::vector<std::string>& filenames
);
