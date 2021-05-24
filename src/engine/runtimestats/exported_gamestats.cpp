#include "exported_gamestats.h"

#include <binary_io.h>

#include <drverror.h>

void GameExportsNodeData::save(std::ostream& out) const {
    write_data(out, FILE_HEADER);
    write_string(out, __TIMESTAMP__);

    // TODO content

    uint32_t subnodeCount = static_cast<uint32_t>(subnodes.size());
    write_data(out, subnodeCount);
    for (const auto& [name, data] : subnodes) {
        write_string(out, name);
        data->save(out);
    }
    write_data(out, FILE_END);
}

void GameExportsNodeData::load(std::istream& in) {
    uint32_t header;
    read_data(in, header);
    drv::drv_assert(header == FILE_HEADER, "Invalid file header");
    std::string stamp;
    read_string(in, stamp);
    if (stamp != __TIMESTAMP__)
        return;

    // TODO content

    uint32_t subnodeCount = 0;
    read_data(in, subnodeCount);
    subnodes.clear();
    for (uint32_t i = 0; i < subnodeCount; ++i) {
        std::string name;
        read_string(in, name);
        std::unique_ptr<GameExportsNodeData> data = std::make_unique<GameExportsNodeData>();
        data->load(in);
    }
    uint32_t ending;
    read_data(in, ending);
    drv::drv_assert(ending == FILE_END, "Invalid file ending");
}
