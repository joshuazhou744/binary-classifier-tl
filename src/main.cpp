#include "loader.h"
#include <tl/tensor.h>

#include <iostream>
#include <vector>
#include <string>

int main() {
  const std::string data_dir = "data/snore_data/";
  const int small_n = -1;

  MelConfig config;

  tl::Tensor x_train;
  std::vector<int> y_train;
  load_train(data_dir + "train_data_wav",
             data_dir + "labels.csv",
             config, small_n, x_train, y_train);


  tl::Tensor x_hold;
  std::vector<std::string> hold_filenames;
  load_holdout(data_dir + "holdout_data_wav",
               config, small_n, x_hold, hold_filenames);

  return 0;
}
