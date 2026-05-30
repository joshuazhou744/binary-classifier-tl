#include "loader.h"
#include <tl/tensor.h>
#include <tl/factory.h>
#include <tl/nn.h>
#include <tl/autograd.h>
#include <tl/optim.h>
#include <tl/loss.h>
#include <tl/model_io.h>

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <random>
#include <algorithm>
#include <cmath>

int main() {
  // build model architecture
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
    &conv1, &bn1, &relu, &pool,
    &conv_dw, &bn_dw, &relu,
    &conv_pw, &bn_pw, &relu, &pool,
    &flat, &fc1, &relu, &drop, &fc2
  });

  // load model
  auto state = model.parameters();
  auto bufs = model.buffers();
  state.insert(state.end(), bufs.begin(), bufs.end());
  tl::load_model("snore_model.tlmd", state);

  // load holdout data
  tl::Tensor x_hold;
  std::vector<std::string> hold_filenames;
  load_holdout(data_dir + "holdout_data_wav", config, small_n, x_hold, hold_filenames);

  return 0;
}

