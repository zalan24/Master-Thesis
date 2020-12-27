#pragma once

#include <string>
#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

class ShaderManager;

template <typename T>
class Option
{
 public:
    Option(const std::string& n, T* val) : name(n), value(val) {}

    Option& operator=(const T& val) {
        *value = val;
        return *this;
    }

    T& get() { return *value; }
    const T& get() const { return *value; }

    const std::string& getName() const { return name; }

 private:
    std::string name;
    T* value;
};

using FloatOption = Option<float>;
using BoolOption = Option<bool>;

class RenderableInterface
{
 public:
    struct RenderContext
    {
        int width;
        int height;
        glm::mat4 pv;
        glm::vec3 lightColor;
        glm::vec3 lightDir;
        glm::vec3 ambientColor;
        const ShaderManager* shaderManager;
    };

    RenderableInterface();
    RenderableInterface(const std::string& name);

    virtual ~RenderableInterface() = 0;

    void render(const RenderContext& context) const {
        if (!isHidden())
            _render(context);
    }

    virtual std::vector<std::string> getPrograms() const { return {}; }

    std::string getName() const { return name; }
    std::string setName(const std::string& n) { return name = n; }

    virtual std::vector<FloatOption> getOptions() { return {}; }
    virtual std::vector<BoolOption> getBoolOptions() { return {}; }

    bool isHidden() const { return hidden; }
    void hide() { hidden = true; }
    void show() { hidden = false; }
    void setHidden(bool h) { hidden = h; }

 protected:
    virtual void _render(const RenderContext& context) const = 0;

 private:
    static std::size_t itemCount;
    bool hidden = false;

    std::string name;
};
