#include "network_builders.h"

NetworkProfile BuildDarkNet19() {
  NetworkProfile net{"darknt19", 64, {}};
  net.blocks.push_back(BlockSpec{"stage1", {
      LayerSpec{"conv1", "op", 0.6000e9, 20.0000},
      LayerSpec{"conv2", "op", 0.9000e9, 16.0000},
  }});
  net.blocks.push_back(BlockSpec{"stage2", {
      LayerSpec{"conv3", "op", 1.4000e9, 14.0000},
      LayerSpec{"conv4", "op", 1.8000e9, 12.0000},
  }});
  net.blocks.push_back(BlockSpec{"stage3", {
      LayerSpec{"conv5", "op", 2.0000e9, 10.0000},
      LayerSpec{"conv6", "op", 2.3000e9, 9.0000},
  }});
  net.blocks.push_back(BlockSpec{"head", {
      LayerSpec{"conv7", "op", 1.2000e9, 6.0000},
      LayerSpec{"pred", "op", 0.5000e9, 3.5000},
  }});
  return net;
}
