#include "network_builders.h"

NetworkProfile BuildAlexNet() {
  NetworkProfile net{"alexnet", 64, {}};
  net.blocks.push_back(BlockSpec{"stem", {
      LayerSpec{"conv1", "op", 0.7200e9, 18.0000},
      LayerSpec{"pool1", "op", 0.1000e9, 9.0000},
  }});
  net.blocks.push_back(BlockSpec{"mid", {
      LayerSpec{"conv2", "op", 1.1000e9, 12.0000},
      LayerSpec{"conv3", "op", 1.3500e9, 10.0000},
  }});
  net.blocks.push_back(BlockSpec{"deep", {
      LayerSpec{"conv4", "op", 1.3000e9, 8.5000},
      LayerSpec{"conv5", "op", 1.2000e9, 8.0000},
  }});
  net.blocks.push_back(BlockSpec{"head", {
      LayerSpec{"fc6", "op", 0.4500e9, 4.0000},
      LayerSpec{"fc7", "op", 0.4500e9, 4.0000},
      LayerSpec{"fc8", "op", 0.0800e9, 1.2000},
  }});
  return net;
}
