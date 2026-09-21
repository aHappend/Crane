// Appended to an isolated copy of pinned SET main.cpp (main renamed), allowing
// reuse of its unmodified Polar core configuration and the actual CoreMapper.
#include "partition.h"
#include <iomanip>

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const std::string name = argv[1];
    auto found = All_Networks.find(name);
    if (found == All_Networks.end()) return 3;
    const Network& net = *found->second;
    const unsigned max_batch = std::stoul(argv[2]);
    const unsigned max_tiles = std::stoul(argv[3]);
    Core* core;
    CoreMapper* mapper;
    init_core("polar", core, mapper);
    PartEngine partitions;
    std::cout << std::setprecision(17);
    for (lid_t id = 0; id < net.len(); ++id) {
        const Node& node = net[id];
        for (unsigned batch=1; batch<=max_batch; batch*=2) {
            for (unsigned tiles=1; tiles<=max_tiles; ++tiles) {
                PartSch partition, best_partition;
                auto iter = partitions.init(tiles, batch, node, partition, 0);
                double best = cost_inf;
                CoreMapper::CoreMapping chosen;
                bool valid = false;
                if (iter) do {
                    auto candidate = mapper->genLayerMap(node.layer(), partition, batch, node.hasWgtPrevs());
                    if (!candidate.cost.is_valid()) continue;
                    double edp = candidate.cost.energy * tiles * candidate.cost.time;
                    if (edp < best) {
                        best = edp; chosen = candidate; best_partition=partition; valid=true;
                    }
                } while (iter.nextPart());
                if (!valid) {
                    std::cout << id << ',' << batch << ',' << tiles << ",invalid\n";
                } else {
                    std::cout << id << ',' << batch << ',' << tiles << ",ok,"
                              << chosen.cost.time << ',' << chosen.cost.energy * tiles << ','
                              << chosen.mac * tiles << ',' << chosen.buffer * tiles << ','
                              << chosen.noc * tiles << ',' << chosen.ubuf * tiles << ','
                              << best_partition.K << ',' << best_partition.B << ','
                              << best_partition.H << ',' << best_partition.W << '\n';
                }
            }
        }
    }
    delete mapper;
    delete core;
}
