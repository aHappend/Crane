#include "network_builders.h"

NetworkProfile BuildGoogleNet() {
  NetworkProfile net{"googlenet", 64, {}};
  net.blocks.push_back(BlockSpec{"stem", {
      LayerSpec{"conv1", "op", 0.9000e9, 18.0000},
      LayerSpec{"pool1", "op", 0.2000e9, 10.0000},
  }});
  net.blocks.push_back(BlockSpec{"inception3", {
      LayerSpec{"inc3a", "op", 2.0000e9, 14.0000},
      LayerSpec{"inc3b", "op", 2.1000e9, 14.0000},
  }});
  net.blocks.push_back(BlockSpec{"inception4", {
      LayerSpec{"inc4a", "op", 2.5000e9, 12.0000},
      LayerSpec{"inc4b", "op", 2.6000e9, 12.0000},
      LayerSpec{"inc4c", "op", 2.6000e9, 12.0000},
  }});
  net.blocks.push_back(BlockSpec{"head", {
      LayerSpec{"inc5", "op", 2.1000e9, 9.0000},
      LayerSpec{"fc", "op", 0.5000e9, 2.0000},
  }});
  return net;
}
