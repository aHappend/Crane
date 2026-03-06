#include "network_builders.h"

NetworkProfile BuildDenseNet() {
  NetworkProfile net{"densenet", 64, {}};
  net.blocks.push_back(BlockSpec{"dense1", {
      LayerSpec{"db1_l1", "op", 1.1000e9, 24.0000},
      LayerSpec{"db1_l2", "op", 1.2000e9, 24.0000},
      LayerSpec{"trans1", "op", 0.4000e9, 18.0000},
  }});
  net.blocks.push_back(BlockSpec{"dense2", {
      LayerSpec{"db2_l1", "op", 1.5000e9, 20.0000},
      LayerSpec{"db2_l2", "op", 1.6000e9, 20.0000},
      LayerSpec{"trans2", "op", 0.5000e9, 15.0000},
  }});
  net.blocks.push_back(BlockSpec{"dense3", {
      LayerSpec{"db3_l1", "op", 1.8000e9, 16.0000},
      LayerSpec{"db3_l2", "op", 2.0000e9, 16.0000},
      LayerSpec{"trans3", "op", 0.6000e9, 12.0000},
  }});
  net.blocks.push_back(BlockSpec{"head", {
      LayerSpec{"classifier", "op", 0.4000e9, 2.0000},
  }});
  return net;
}
