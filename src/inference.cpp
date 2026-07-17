#include "loader.h"
#include "model.h"
#include <tl/tensor.h>
#include <tl/factory.h>
#include <tl/autograd.h>
#include <tl/ops.h>
#include <tl/model_io.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>

std::filesystem::create_directories("outputs");

int main() {
  const std::string data_dir = "data/snore_data/";
  const int small_n = -1;

  MelConfig config;

  // load holdout data
  tl::Tensor x_hold;
  std::vector<std::string> hold_filenames;
  load_holdout(data_dir + "holdout_data_wav", config, small_n, x_hold, hold_filenames);

  const int N = (int) x_hold.sizes()[0];
  const int64_t n_mels = x_hold.sizes()[2];
  const int64_t n_frames = x_hold.sizes()[3];
  const int64_t per_sample = 1 * n_mels * n_frames;
  std::cout << "loaded " << N << " samples, 1x" << n_mels << "x" << n_frames << "\n";


  // load model
  SnoreModel model;
  auto state = model.parameters();
  auto bufs = model.buffers();
  state.insert(state.end(), bufs.begin(), bufs.end());
  tl::load_model("outputs/snore_model.tlmd", state);
  model.eval();

  const int batch_size = 8;
  tl::Tensor x_batch = tl::zeros({batch_size, 1, n_mels, n_frames});
  std::ofstream out("outputs/predictions.csv");
  out << "filename,snore_probability\n";

  tl::NoGradGuard no_grad;

  for (int start = 0; start + batch_size <= N; start += batch_size) {
    const int n_valid = std::min(batch_size, N - start);
    for (int b = 0; b < n_valid; ++b) {
      const float* sp = x_hold.data() + (int64_t)(start + b) * per_sample;
      float* dp = x_batch.data() + (int64_t) b * per_sample;
      std::copy(sp, sp + per_sample, dp);
    }

    tl::Tensor logits = model.forward(x_batch);
    tl::Tensor probs = tl::softmax(logits);

    for (int b = 0; b < n_valid; ++b) {
      float snore_prob = probs.data()[b*2 + 1]; // P(class 1)
      out << hold_filenames[start + b] << "," << snore_prob << "\n";
    }
  }

  std::cout << "wrote to predictions.csv\n";

  return 0;
}

