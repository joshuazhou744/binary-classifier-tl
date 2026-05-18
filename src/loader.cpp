#include "loader.h"

#include <tl/tensor.h>
#include <tl/factory.h>
#include <tl/audio.h>
#include <tl/ops.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <dirent.h>

// helper: build "<dir>/<file>" path
static std::string join_path(const std::string& dir, const std::string& file) {
  if (!dir.empty() && dir.back() == '/') return dir + file;
  return dir + "/" + file;
}

tl::Tensor compute_log_mel(const std::vector<float>& samples, const MelConfig& config) {
  // pad or truncate to fixed length
  int64_t fixed_length = static_cast<int64_t>(config.sample_rate) * static_cast<int64_t>(config.clip_duration);
  std::vector<float> y(fixed_length, 0.0f);
  int64_t n_copy = std::min<int64_t>(static_cast<int64_t>(samples.size()), fixed_length);
  std::copy(samples.begin(), samples.begin() + n_copy, y.begin());

  // raw mel-spec: (1, frames, n_mels) n_mels == mel_bins
  tl::Tensor mel = tl::audio::mel_spectrogram(
      y, config.sample_rate, config.n_fft, config.hop_length, config.n_mels
  );
  // reshape to (1, n_mels, frames)
  mel = tl::transpose(mel, 1, 2);

  // power (amplitude) -> dB, floored at db_floor
  float* data = mel.data();
  int64_t n = mel.numel();

  float max_val = 1e-10f;
  for (int64_t i = 0; i < n; ++i) {
    if (data[i] > max_val) max_val = data[i];
  }
  for (int64_t i = 0; i < n; ++i) {
    float ratio = data[i] / max_val;
    if (ratio <= 0.0f) {
      data[i] = config.db_floor;
    } else {
       float db = 10.0f * std::log10(ratio);
       data[i] = std::max(db, config.db_floor);
    }
  }
  return mel;
}

std::vector<std::pair<std::string, int>> read_labels(const std::string& csv_path) {
  std::ifstream f(csv_path);
  if (!f) throw std::runtime_error("read_labels: cannot open " + csv_path);

  std::vector<std::pair<std::string, int>> out;
  std::string line;
  bool first = true;
  while (std::getline(f, line)) {
    if (first) { first = false; continue; } // skip header line
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;

    auto comma = line.find(',');
    // no comma found
    if (comma == std::string::npos) {
      throw std::runtime_error("read_labels: malformed line: " + line);
    }
    std::string file_id = line.substr(0, comma);
    int label = std::stoi(line.substr(comma+1));
    // emplace and move file_id into the output vector rather than copy
    out.emplace_back(std::move(file_id), label);
  }
  return out;
}

void load_train(
    const std::string& wav_dir,
    const std::string& labels_csv,
    const MelConfig& config,
    int max_count,
    tl::Tensor& x,
    std::vector<int>& labels
) {
  auto pairs = read_labels(labels_csv);
  if (pairs.empty()) throw std::runtime_error("load_train: no labels read");
  if (max_count > 0 && static_cast<int>(pairs.size()) > max_count) {
    pairs.resize(max_count);
  }

  int N = static_cast<int>(pairs.size());
  std::cout << "load_train: processing " << N << " samples\n";

  // read first sample to determine output shape
  auto w0 = tl::audio::load_wav(join_path(wav_dir, pairs[0].first));
  tl::Tensor mel0 = compute_log_mel(w0.samples, config);
  int64_t n_mels = mel0.sizes()[1];
  int64_t n_frames = mel0.sizes()[2];
  int64_t per_sample = n_mels * n_frames;

  x = tl::zeros({N, 1, n_mels, n_frames});
  labels.resize(N);

  std::memcpy(x.data(), mel0.data(), per_sample * sizeof(float));
  labels[0] = pairs[0].second;

  for (int i = 1; i < N; ++i) {
    auto w = tl::audio::load_wav(join_path(wav_dir, pairs[i].first));
    tl::Tensor mel = compute_log_mel(w.samples, config);
    std::memcpy(x.data() + i * per_sample, mel.data(), per_sample * sizeof(float));
    labels[i] = pairs[i].second;

    if ((i + 1) % 100 == 0) {
      std::cout << (i+1) << "/" << N << "\n";
    }
  }
}

void load_holdout(
    const std::string& wav_dir,
    const MelConfig& config,
    int max_count,
    tl::Tensor& x,
    std::vector<std::string>& filenames
) {
  // list .wav files in wav_dir
  filenames.clear();
  DIR* dir = opendir(wav_dir.c_str());
  if (!dir) throw std::runtime_error("load_holdout: cannot open dir " + wav_dir);
  while (auto* ent = readdir(dir)) {
    std::string name = ent->d_name;
    if (name.size() >= 4 && name.substr(name.size() - 4) == ".wav") {
      filenames.push_back(std::move(name));
    }
  }
  closedir(dir);
  std::sort(filenames.begin(), filenames.end());

  if (filenames.empty()) throw std::runtime_error("load_holdout: no .wav files in " + wav_dir);
  if (max_count > 0 && static_cast<int>(filenames.size()) > max_count) {
    filenames.resize(max_count);
  }

  int N = static_cast<int>(filenames.size());
  std::cout << "load_holdout: processing " << N << " samples\n";

  auto w0 = tl::audio::load_wav(join_path(wav_dir, filenames[0]));
  tl::Tensor mel0 = compute_log_mel(w0.samples, config);
  int64_t n_mels = mel0.sizes()[1];
  int64_t n_frames = mel0.sizes()[2];
  int64_t per_sample = n_mels * n_frames;

  x = tl::zeros({N, 1, n_mels, n_frames});
  std::memcpy(x.data(), mel0.data(), per_sample * sizeof(float));

  for (int i = 1; i < N; ++i) {
    auto w = tl::audio::load_wav(join_path(wav_dir, filenames[i]));
    tl::Tensor mel = compute_log_mel(w.samples, config);
    std::memcpy(x.data() + i * per_sample, mel.data(), per_sample * sizeof(float));

    if ((i + 1) % 50 == 0) {
      std::cout << (i+1) << "/" << N << "\n";
    }
  }
}
