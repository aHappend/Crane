#include "network_builders.h"

NetworkProfile BuildGNMT() {
  NetworkProfile net{"gnmt", 128, {}};
  net.blocks.push_back(BlockSpec{"enc_embed", {
      LayerSpec{"src_embed", "op", 0.8000e9, 28.0000},
  }});
  net.blocks.push_back(BlockSpec{"encoder", {
      LayerSpec{"enc_lstm1", "op", 3.8000e9, 30.0000},
      LayerSpec{"enc_lstm2", "op", 3.6000e9, 30.0000},
      LayerSpec{"enc_lstm3", "op", 3.5000e9, 30.0000},
  }});
  net.blocks.push_back(BlockSpec{"decoder", {
      LayerSpec{"dec_lstm1", "op", 4.0000e9, 32.0000},
      LayerSpec{"dec_lstm2", "op", 3.7000e9, 32.0000},
  }});
  net.blocks.push_back(BlockSpec{"attn_head", {
      LayerSpec{"attn", "op", 1.6000e9, 24.0000},
      LayerSpec{"softmax", "op", 0.7000e9, 8.0000},
  }});
  return net;
}
