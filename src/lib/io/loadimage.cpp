#include "loadimage.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

unsigned char* load_image(const std::string& filename, int& width, int& height, int& channels) {
    return stbi_load(filename.c_str(), &width, &height, &channels, 0);
}

void free_image(unsigned char* img) {
    stbi_image_free(img);
}
