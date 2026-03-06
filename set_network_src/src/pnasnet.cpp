#include "network_builders.h"

NetworkProfile BuildPNASNet() {
  NetworkProfile net{"pnasnet", 64, {}};
  net.blocks.push_back(BlockSpec{"stem", {
      LayerSpec{"conv1", "op", 0.9000e9, 18.0000},
  }});
  net.blocks.push_back(BlockSpec{"cell1", {
      LayerSpec{"sepconv1", "op", 2.4000e9, 14.0000},
      LayerSpec{"sepconv2", "op", 2.5000e9, 14.0000},
  }});
  net.blocks.push_back(BlockSpec{"cell2", {
      LayerSpec{"sepconv3", "op", 2.8000e9, 12.0000},
      LayerSpec{"sepconv4", "op", 2.9000e9, 12.0000},
  }});
  net.blocks.push_back(BlockSpec{"cell3", {
      LayerSpec{"sepconv5", "op", 3.1000e9, 10.0000},
      LayerSpec{"pool_proj", "op", 0.8000e9, 8.0000},
  }});
  net.blocks.push_back(BlockSpec{"head", {
      LayerSpec{"fc", "op", 0.5000e9, 2.0000},
  }});
  return net;
}
