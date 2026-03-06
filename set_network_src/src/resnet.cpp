#include "network_builders.h"

NetworkProfile BuildResNet() {
  NetworkProfile net{"resnet", 64, {}};
  net.blocks.push_back(BlockSpec{"stem", {
      LayerSpec{"conv1", "op", 1.0000e9, 20.0000},
      LayerSpec{"pool1", "op", 0.2000e9, 12.0000},
  }});
  net.blocks.push_back(BlockSpec{"stage2", {
      LayerSpec{"res2_1", "op", 3.0000e9, 16.0000},
      LayerSpec{"res2_2", "op", 2.8000e9, 16.0000},
  }});
  net.blocks.push_back(BlockSpec{"stage3", {
      LayerSpec{"res3_1", "op", 4.2000e9, 14.0000},
      LayerSpec{"res3_2", "op", 4.0000e9, 14.0000},
  }});
  net.blocks.push_back(BlockSpec{"stage4", {
      LayerSpec{"res4_1", "op", 5.0000e9, 12.0000},
      LayerSpec{"res4_2", "op", 4.8000e9, 12.0000},
  }});
  net.blocks.push_back(BlockSpec{"head", {
      LayerSpec{"fc", "op", 0.6000e9, 2.2000},
  }});
  return net;
}
