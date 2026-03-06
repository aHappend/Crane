#include "network_builders.h"

NetworkProfile BuildZFNet() {
  NetworkProfile net{"zfnet", 64, {}};
  net.blocks.push_back(BlockSpec{"stem", {
      LayerSpec{"conv1", "op", 0.6000e9, 16.0000},
      LayerSpec{"pool1", "op", 0.1000e9, 8.0000},
  }});
  net.blocks.push_back(BlockSpec{"mid", {
      LayerSpec{"conv2", "op", 1.0000e9, 12.0000},
      LayerSpec{"conv3", "op", 1.2000e9, 10.0000},
  }});
  net.blocks.push_back(BlockSpec{"deep", {
      LayerSpec{"conv4", "op", 1.2000e9, 8.0000},
      LayerSpec{"conv5", "op", 1.1000e9, 7.0000},
  }});
  net.blocks.push_back(BlockSpec{"head", {
      LayerSpec{"fc6", "op", 0.4000e9, 3.0000},
      LayerSpec{"fc7", "op", 0.4000e9, 3.0000},
      LayerSpec{"fc8", "op", 0.1000e9, 1.0000},
  }});
  return net;
}
