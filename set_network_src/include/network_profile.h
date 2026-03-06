#pragma once

#include <string>
#include <vector>

struct LayerSpec {
  std::string name;
  std::string type;
  double flops;
  double output_mb;
};

struct BlockSpec {
  std::string name;
  std::vector<LayerSpec> layers;
};

struct NetworkProfile {
  std::string name;
  int batch_size;
  std::vector<BlockSpec> blocks;
};
