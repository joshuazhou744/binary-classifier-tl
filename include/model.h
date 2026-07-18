#pragma once
#include <tl/nn.h>
#include <tl/tensor.h>
#include <vector>

struct SnoreModel {
  tl::nn::InputNormalize norm;
  tl::nn::Conv2d conv1{1, 16, 3, 1, 1};
  tl::nn::BatchNorm2d bn1{16};
  tl::nn::Conv2d conv_dw{16, 16, 3, 1, 1, 16};
  tl::nn::BatchNorm2d bn_dw{16};
  tl::nn::Conv2d conv_pw{16, 32, 1};
  tl::nn::BatchNorm2d bn_pw{32};
  tl::nn::Linear fc1{32, 64};
  tl::nn::Dropout drop{0.3f};
  tl::nn::Linear fc2{64, 2};
  tl::nn::ReLU relu;
  tl::nn::MaxPool2d pool{2, 2};
  tl::nn::GlobalAvgPool2d gap;

  tl::nn::Sequential net{{
    &norm,
    &conv1, &bn1, &relu, &pool,
    &conv_dw, &bn_dw, &relu,
    &conv_pw, &bn_pw, &relu, &pool,
    &gap, &fc1, &relu, &drop, &fc2
  }};

  // forward all methods to net
  tl::Tensor forward(const tl::Tensor& x) const { return net.forward(x); }
  std::vector<tl::Tensor*> parameters() { return net.parameters(); }
  std::vector<tl::Tensor*> buffers() { return net.buffers(); }
  void train() { net.train(); }
  void eval() { net.eval(); }

  // disallow copying SnoreModel because net holds pointers to members
  SnoreModel() = default;
  SnoreModel(const SnoreModel&) = delete;
  SnoreModel& operator=(const SnoreModel&) = delete;
};
