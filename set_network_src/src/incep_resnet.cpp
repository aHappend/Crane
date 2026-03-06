#include "network_builders.h"

NetworkProfile BuildIncepResNet() {
  NetworkProfile net{"incep_resnet", 64, {}};
  net.blocks.push_back(BlockSpec{"stem", {
      LayerSpec{"conv_stem", "op", 1.1000e9, 20.0000},
  }});
  net.blocks.push_back(BlockSpec{"block_a", {
      LayerSpec{"ir_a1", "op", 3.0000e9, 16.0000},
      LayerSpec{"ir_a2", "op", 3.0000e9, 16.0000},
  }});
  net.blocks.push_back(BlockSpec{"block_b", {
      LayerSpec{"ir_b1", "op", 3.5000e9, 14.0000},
      LayerSpec{"ir_b2", "op", 3.6000e9, 14.0000},
  }});
  net.blocks.push_back(BlockSpec{"block_c", {
      LayerSpec{"ir_c1", "op", 4.0000e9, 12.0000},
      LayerSpec{"ir_c2", "op", 4.1000e9, 12.0000},
      LayerSpec{"fc", "op", 0.7000e9, 2.4000},
  }});
  return net;
}
