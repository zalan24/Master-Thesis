#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

class FileManager
{
 public:
    static FileManager* getSingleton() { return instance; }

    explicit FileManager(const std::string& folderName);
    ~FileManager();

    FileManager(const FileManager&) = delete;
    FileManager& operator=(const FileManager&) = delete;

    std::ifstream readCache(const std::string& name) const;
    std::ofstream writeCache(const std::string& name) const;

    struct Directory
    {
        fs::path path;
        std::vector<Directory> subDirs;
        std::vector<fs::path> files;
    };

    const Directory& getRoot() const { return root; }

 private:
    static FileManager* instance;

    fs::path folderPath;
    Directory root;

    Directory parse(const fs::path& path) const;

    void createCacheDir() const;

    std::string getCacheFolderName() const { return ".cache"; }
};
