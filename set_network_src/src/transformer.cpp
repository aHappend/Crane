#include "network_builders.h"

NetworkProfile BuildTransformer() {
  NetworkProfile net{"transformer", 128, {}};
  net.blocks.push_back(BlockSpec{"embed", {
      LayerSpec{"embed", "op", 1.7000e9, 38.0000},
  }});
  net.blocks.push_back(BlockSpec{"enc1", {
      LayerSpec{"attn1", "op", 4.0000e9, 40.0000},
      LayerSpec{"mlp1", "op", 6.4000e9, 43.0000},
  }});
  net.blocks.push_back(BlockSpec{"enc2", {
      LayerSpec{"attn2", "op", 3.9000e9, 40.0000},
      LayerSpec{"mlp2", "op", 6.2000e9, 43.0000},
  }});
  net.blocks.push_back(BlockSpec{"enc3", {
      LayerSpec{"attn3", "op", 3.8000e9, 40.0000},
      LayerSpec{"mlp3", "op", 6.0000e9, 42.0000},
  }});
  net.blocks.push_back(BlockSpec{"head", {
      LayerSpec{"proj", "op", 1.9000e9, 14.0000},
  }});
  return net;
}
