#include "loader.h"
#include <tl/tensor.h>
#include <tl/factory.h>
#include <tl/nn.h>
#include <tl/autograd.h>
#include <tl/model_io.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

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

  auto pool_out = [](int64_t L) { return (L - 2) / 2 + 1; };
  int64_t Hf = pool_out(pool_out(n_mels));
  int64_t Wf = pool_out(pool_out(n_frames));
  int64_t flattened = 32 * Hf * Wf;

  // build model architecture
  tl::nn::InputNormalize norm;
  tl::nn::Conv2d conv1(1, 16, 3, 1, 1); // (in, out, kernel, stride, pad)
  tl::nn::BatchNorm2d bn1(16);
  // depthwise-seperable block: dw conv -> bn -> relu, pw conv -> bn -> relu -> pool
  tl::nn::Conv2d conv_dw(16, 16, 3, 1, 1, 16);
  tl::nn::BatchNorm2d bn_dw(16);
  tl::nn::Conv2d conv_pw(16, 32, 1);
  tl::nn::BatchNorm2d bn_pw(32);
  // head: flatten -> fc1 -> relu -> dropout -> fc2
  tl::nn::Flatten flat;
  tl::nn::Linear fc1(flattened, 64);
  tl::nn::Dropout drop(0.5f);
  tl::nn::Linear fc2(64, 2); // project finally to 2 classes (logits)

  // stateless ops, reused
  tl::nn::ReLU relu;
  tl::nn::MaxPool2d pool(2, 2);

  tl::nn::Sequential model({
    &norm,
    &conv1, &bn1, &relu, &pool,
    &conv_dw, &bn_dw, &relu,
    &conv_pw, &bn_pw, &relu, &pool,
    &flat, &fc1, &relu, &drop, &fc2
  });

  // load model
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
    for (int b = 0; b < batch_size; ++b) {
      const float* sp = x_hold.data() + (int64_t)(start + b) * per_sample;
      float* dp = x_batch.data() + (int64_t) b * per_sample;
      std::copy(sp, sp + per_sample, dp);
    }

    tl::Tensor logits = model.forward(x_batch);
    tl::Tensor probs = tl::softmax(logits);

    for (int b = 0; b < batch_size; ++b) {
      float snore_prob = probs.data()[b*2 + 1]; // P(class 1)
      out << hold_filenames[start + b] << "," << snore_prob << "\n";
    }
  }

  std::cout << "wrote to predictions.csv\n";

  return 0;
}

