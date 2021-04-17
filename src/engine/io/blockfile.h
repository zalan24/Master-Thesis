#pragma once

#include <string>
#include <unordered_map>
#include <variant>

class BlockFile
{
 public:
    BlockFile(std::istream& in, bool contains_comments = true);

    bool hasContent() const;
    bool hasNodes() const;

    size_t getBlockCount() const;
    const BlockFile* getNode(size_t index) const;
    const std::string& getBlockName(size_t index) const;
    size_t getBlockCount(const std::string& name) const;
    const BlockFile* getNode(const std::string& name, size_t index = 0) const;

    const std::string* getContent() const;

 private:
    using Map = std::unordered_multimap<std::string, BlockFile>;
    std::variant<Map, std::string> content;
};
