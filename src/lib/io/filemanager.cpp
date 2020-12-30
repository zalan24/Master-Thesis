#include "filemanager.h"

#include <cassert>

#include <base64.h>

FileManager* FileManager::instance = nullptr;

FileManager::FileManager(const std::string& folderName) : folderPath(folderName) {
    assert(instance == nullptr);
    instance = this;
    root = parse(folderPath);
}

FileManager::~FileManager() {
    instance = this;
}

FileManager::Directory FileManager::parse(const fs::path& path) const {
    Directory ret;
    ret.path = path;
    for (const auto& entry : fs::directory_iterator(path)) {
        fs::path e = entry;
        if (fs::is_directory(e)) {
            ret.subDirs.push_back(parse(e));
        }
        else if (fs::is_regular_file(e)) {
            ret.files.push_back(e);
        }
    }
    return ret;
}

static std::string getFileName(const std::string& name) {
    return base64_encode(reinterpret_cast<const unsigned char*>(name.c_str()), name.length());
}

static void createDir(const std::string& pathStr) {
    fs::path path{pathStr};
    if (!fs::exists(path)) {
        if (!fs::create_directory(path)) {
            throw std::runtime_error("Could not create directory: " + path.string());
        }
    }
}

void FileManager::createCacheDir() const {
    createDir(getCacheFolderName());
}

std::ifstream FileManager::readCache(const std::string& name) const {
    createCacheDir();
    fs::path path{getCacheFolderName()};
    path /= getFileName(name);
    path += ".bin";
    std::ifstream ret(path.string().c_str(), std::ios::binary);
    return ret;
}

std::ofstream FileManager::writeCache(const std::string& name) const {
    createCacheDir();
    fs::path path{getCacheFolderName()};
    path /= getFileName(name);
    path += ".bin";
    std::ofstream ret(path.string().c_str(), std::ios::binary);
    return ret;
}
