// Extract factual workload metadata from a separately checked-out SET revision.
// Build instructions are implemented by import_set_networks.py.
#include "network.h"
#include "nns/nns.h"
#include "json/json.h"
#include <iostream>
#include <map>
#include <string>

static Json::Value shape(const fmap_shape& value) {
    Json::Value out(Json::arrayValue);
    out.append(static_cast<double>(value.c));
    out.append(static_cast<double>(value.h));
    out.append(static_cast<double>(value.w));
    return out;
}

static Json::Value indices(const Bitset& values) {
    Json::Value out(Json::arrayValue);
    FOR_BITSET(i, values) { out.append(static_cast<double>(i)); }
    return out;
}

int main() {
    const std::map<std::string, const Network*> networks = {
        {"alexnet", &alexnet}, {"resnet50", &resnet50}, {"resnet101", &resnet101},
        {"vgg19", &vgg19}, {"googlenet", &googlenet}, {"densenet", &densenet},
        {"darknet19", &darknet19}, {"zfnet", &zfnet}, {"gnmt", &gnmt},
        {"inception_resnet_v1", &inception_resnet_v1}, {"pnasnet", &PNASNet},
        {"transformer", &transformer}, {"transformer_cell", &transformer_cell},
        {"bert_large_cell", &BERT_block}, {"gpt2_xl_prefill_cell", &GPT2_prefill_block},
        {"gpt2_xl_decode_cell", &GPT2_decode_block}
    };
    Json::Value root(Json::objectValue);
    for (const auto& item : networks) {
        Json::Value layers(Json::arrayValue);
        for (lid_t id = 0; id < item.second->len(); ++id) {
            const Node& node = (*item.second)[id];
            const Layer& layer = node.layer();
            const auto* conv = dynamic_cast<const ConvLayer*>(&layer);
            const auto* group = dynamic_cast<const GroupConvLayer*>(&layer);
            const auto* fc = dynamic_cast<const FCLayer*>(&layer);
            Json::Value row(Json::objectValue);
            row["id"] = static_cast<double>(id);
            row["name"] = node.name();
            row["op_type"] = group ? "group_conv" : fc ? "linear" : conv ? "conv2d" :
                dynamic_cast<const EltwiseLayer*>(&layer) ? "elementwise" :
                dynamic_cast<const PoolingLayer*>(&layer) ? "pooling" : "local";
            row["input_shape"] = shape(layer.real_ifmap_shape());
            row["output_shape"] = shape(layer.ofmap_shape());
            row["weight_elements"] = static_cast<double>(layer.weight_size());
            row["upstream_operations_per_sample"] = static_cast<double>(layer.get_num_op(1));
            // SET counts convolution multiply-accumulates as one operation.
            row["operations_per_sample"] = static_cast<double>(layer.get_num_op(1) * (conv ? 2 : 1));
            row["activation_parents"] = indices(node.getIfmPrevs());
            row["weight_parents"] = indices(node.getWgtPrevs());
            row["parents"] = indices(node.getPrevs());
            row["external_input_channels"] = static_cast<double>(node.get_external_C());
            if (conv) {
                const auto& wl = conv->get_workload();
                row["kernel_h"] = static_cast<double>(wl.R);
                row["kernel_w"] = static_cast<double>(wl.S);
                row["stride_h"] = static_cast<double>(wl.sH);
                row["stride_w"] = static_cast<double>(wl.sW);
                row["groups"] = static_cast<double>(group ? group->get_workload().G : 1);
            }
            layers.append(row);
        }
        root[item.first] = layers;
    }
    Json::StyledWriter writer;
    std::cout << writer.write(root);
}
