#include "loader.h"
#include <tl/tensor.h>
#include <tl/factory.h>
#include <tl/nn.h>
#include <tl/autograd.h>
#include <tl/optim.h>
#include <tl/loss.h>

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <random>
#include <algorithm>
#include <cmath>

// copy batch_size samples into a contiguous batch tensor
static void make_batch(const tl::Tensor& x_all, const std::vector<int>& y_all, const std::vector<int>& index, int start, int batch_size, int64_t per_sample, tl::Tensor& x_batch, std::vector<int>& y_batch) {
  for (int b = 0; b < batch_size; ++b) {
    int src = index[start + b];
    const float* sp = x_all.data() + (int64_t) src * per_sample;
    float* dp = x_batch.data() + (int64_t) b * per_sample;
    std::copy(sp, sp + per_sample, dp);
    y_batch[b] = y_all[src];
  }
}

int main() {
  const std::string data_dir = "data/snore_data/";
  const int small_n = -1;

  MelConfig config;

  std::cout << "loading training data\n";
  tl::Tensor x_all;            // full labeled pool (train + val)
  std::vector<int> y_all;
  load_train(data_dir + "train_data_wav",
             data_dir + "labels.csv",
             config, small_n, x_all, y_all);

  const int N = (int) y_all.size();
  const int64_t n_mels = x_all.sizes()[2];
  const int64_t n_frames = x_all.sizes()[3];
  const int64_t per_sample = 1 * n_mels * n_frames;
  std::cout << "loaded " << N << " samples, 1x" << n_mels << "x" << n_frames << "\n";

  // 80/20 train/val split
  std::mt19937 rng(41);
  std::vector<int> i0, i1;
  for (int i =0; i < N; ++i) {
    if (y_all[i] == 0) {
      i0.push_back(i);
    } else {
      i1.push_back(i);
    }
  }
  std::shuffle(i0.begin(), i0.end(), rng);
  std::shuffle(i1.begin(), i1.end(), rng);

  std::vector<int> train_i, val_i;
  auto split = [&](const std::vector<int>& src) {
    int n_val = (int)(src.size() * 0.2); // 20% to validation
    for (int i = 0; i < (int) src.size(); ++i) {
      (i < n_val ? val_i : train_i).push_back(src[i]);
    }
  };
  split(i0);
  split(i1);
  std::cout << "split -> train: " << train_i.size() << " val: " << val_i.size() << "\n";

  // global mean/std on train data only
  double sum = 0.0, sq = 0.0;
  for (int s: train_i) {
    const float* p = x_all.data() + (int64_t)s * per_sample;
    for (int64_t i = 0; i < per_sample; ++i) {
      sum += p[i];
      sq += (double) p[i] * p[i];
    }
  }
  int64_t total = (int64_t) train_i.size() * per_sample;
  float mean = (float)(sum / total);
  float standard_dev = (float)std::sqrt(sq / total - (double)mean*mean) + 1e-6f;
  std::cout << "norm: mean=" << mean << " standard deviation=" << standard_dev << "\n";

  // use train data stats (mean/std) for all labeled data
  float* d = x_all.data();
  for (int64_t i = 0, n = x_all.numel(); i < n; ++i) {
    d[i] = (d[i] - mean) / standard_dev;
  }

  std::cout << "data ready\n";

  // model
  auto pool_out = [](int64_t L) {
    return (L - 2) / 2 + 1;
  };
  int64_t Hf = pool_out(pool_out(n_mels)); // 64 -> 32 -> 16
  int64_t Wf = pool_out(pool_out(n_frames)); // 469 -> 234 -> 117
  int64_t flattened = 32 * Hf * Wf;
  std::cout << "flattened=" << flattened << "\n";

  // block 1: conv -> bn -> relu -> pool
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

  int64_t pcount = 0;
  for (auto* p: model.parameters()) pcount += p->numel();
  std::cout << "model params: " << pcount << "\n";

  // trainin
  tl::Adam opt(model.parameters(), 1e-4f, 1e-5, 0.9f, 0.999f, 1e-8f);
  const int batch_size = 16;
  const int epochs = 5;

  tl::Tensor x_batch = tl::zeros({batch_size, 1, n_mels, n_frames});
  std::vector<int> y_batch(batch_size);
  std::mt19937 shuffle_rng(71);

  for (int epoch = 0; epoch < epochs; ++epoch) {
    // train mode
    drop.set_training(true);
    std::shuffle(train_i.begin(), train_i.end(), shuffle_rng);

    float running = 0.0f;
    int n_batches = 0;
    const int n_tr = (int) train_i.size();
    for (int start = 0; start + batch_size <= n_tr; start += batch_size) {
      make_batch(x_all, y_all, train_i, start, batch_size, per_sample, x_batch, y_batch);

      tl::Tensor logits = model.forward(x_batch);
      tl::Tensor loss = tl::cross_entropy_loss(logits, y_batch);

      opt.zero_grad();
      loss.backward();
      opt.step();
      tl::release_graph(loss);

      running += loss.data()[0];
      ++n_batches;
      std::cout << " epoch " << (epoch+1) << " batch " << n_batches
                << " loss=" << loss.data()[0] << "\n";
    }
    float avg_loss = running / n_batches;

    // validate
    drop.set_training(false);
    int correct = 0, errors = 0, total_train = 0;
    const int n_va = (int) val_i.size();
    for (int start = 0; start + batch_size <= n_va; start += batch_size) {
      make_batch(x_all, y_all, val_i, start, batch_size, per_sample, x_batch, y_batch);
      tl::Tensor logits = model.forward(x_batch);
      for (int b = 0; b < batch_size; ++b) {
        int pred = logits.data()[b*2 + 1] > logits.data()[b*2 + 0] ? 1 : 0;
        if (pred == y_batch[b]) ++correct; else ++errors;
        ++total_train;
      }
      tl::release_graph(logits);
    }

    float acc = 100.0f * correct / total_train;
    float mae = (float) errors / total_train;

    std::cout << "epoch " << (epoch+1) << "/" << epochs
              << " loss=" << avg_loss
              << " val_acc=" << acc << "%"
              << " mae=" << mae << "\n";

  }

  tl::Tensor x_hold;
  std::vector<std::string> hold_filenames;
  load_holdout(data_dir + "holdout_data_wav",
               config, small_n, x_hold, hold_filenames);

  return 0;
}
