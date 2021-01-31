#include "blockfile.h"

#include <cctype>
#include <iostream>
#include <sstream>

#include "uncomment.h"

static bool get_node(std::istream& in, std::stringstream& name, std::ostream& out) {
    char c;
    while (std::isspace(in.peek()) && in.get(c))
        out.put(c);
    bool foundName = false;
    while (std::isalnum(in.peek()) && in.get(c)) {
        name.put(c);
        out.put(c);
        foundName = true;
    }
    if (!foundName)
        return false;
    while (std::isspace(in.peek()) && in.get(c))
        out.put(c);
    if (in.peek() == '{' && in.get(c)) {
        out.put(c);
        return true;
    }
    return false;
}

static bool is_empty(std::istream& in) {
    char c;
    while (in.get(c)) {
        if (c == '}')
            return true;
        if (!std::isspace(c))
            return false;
    }
    return true;
}

BlockFile::BlockFile(std::istream& inStream, bool contains_comments) {
    std::stringstream ss;
    if (contains_comments)
        uncomment(inStream, ss);
    std::istream& in = contains_comments ? ss : inStream;

    bool inString = false;
    bool inChar = false;
    char lastC = '\0';
    char c;
    int depth = 0;

    std::stringstream contentSS;
    Map nodes;
    std::stringstream nodeName;
    while (get_node(in, nodeName, contentSS)) {
        nodes.insert({nodeName.str(), BlockFile(in, false)});
        nodeName = std::stringstream();
    }
    if (nodes.size() > 0) {
        if (!is_empty(in))
            throw std::runtime_error("A BlockFile cannot have both string content and other nodes");
        content = std::move(nodes);
    }
    else {
        std::stringstream processedSS;
        while (contentSS.get(c) || in.get(c)) {
            switch (c) {
                case '\'':
                    if (!inString && (!inChar || lastC != '\\'))
                        inChar = !inChar;
                    break;
                case '"':
                    if (!inChar && (!inString || lastC != '\\'))
                        inString = !inString;
                    break;
                case '{':
                    if (!inChar && !inString)
                        depth++;
                    break;
                case '}':
                    if (!inChar && !inString)
                        depth--;
                    break;
                default:
                    break;
            }
            if (depth < 0)
                break;
            processedSS << c;
            lastC = c;
        }
        content = processedSS.str();
    }
}

bool BlockFile::hasContent() const {
    return std::holds_alternative<std::string>(content);
}

bool BlockFile::hasNodes() const {
    return std::holds_alternative<Map>(content);
}

size_t BlockFile::getBlockCount(const std::string& name) const {
    if (!hasNodes())
        throw std::runtime_error("This BlockFile doesn't hold any blocks");
    const Map& m = std::get<Map>(content);
    return m.count(name);
}

const BlockFile* BlockFile::getNode(const std::string& name, size_t index) const {
    if (!hasNodes())
        throw std::runtime_error("This BlockFile doesn't hold any blocks");
    if (getBlockCount(name) <= index)
        throw std::runtime_error("index out of range");
    const Map& m = std::get<Map>(content);
    auto range = m.equal_range(name);
    return &std::next(range.first, static_cast<decltype(range.first)::difference_type>(index))
              ->second;
}

const std::string* BlockFile::getContent() const {
    if (!hasContent())
        throw std::runtime_error("This BlockFile has no content");
    return &std::get<std::string>(content);
}
