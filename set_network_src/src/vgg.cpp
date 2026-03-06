#include "network_builders.h"

NetworkProfile BuildVGG() {
  NetworkProfile net{"vgg", 64, {}};
  net.blocks.push_back(BlockSpec{"conv12", {
      LayerSpec{"conv1", "op", 0.8000e9, 22.0000},
      LayerSpec{"conv2", "op", 1.2000e9, 20.0000},
  }});
  net.blocks.push_back(BlockSpec{"conv34", {
      LayerSpec{"conv3", "op", 1.8000e9, 16.0000},
      LayerSpec{"conv4", "op", 2.1000e9, 16.0000},
  }});
  net.blocks.push_back(BlockSpec{"conv56", {
      LayerSpec{"conv5", "op", 2.5000e9, 13.0000},
      LayerSpec{"conv6", "op", 2.8000e9, 13.0000},
  }});
  net.blocks.push_back(BlockSpec{"head", {
      LayerSpec{"fc6", "op", 0.9000e9, 6.0000},
      LayerSpec{"fc7", "op", 0.9000e9, 6.0000},
      LayerSpec{"fc8", "op", 0.2000e9, 2.0000},
  }});
  return net;
}
