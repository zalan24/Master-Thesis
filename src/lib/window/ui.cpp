#include "ui.h"

#include <iostream>

#include "filemanager.h"
#include "loadmesh.h"
#include "mesh.h"
#include "meshmanager.h"
#include "sampler.h"
#include "shadermanager.h"
#include "spheremesh.h"

#include <GLFW/glfw3.h>

#include "imgui.h"

#include "examples/imgui_impl_glfw.h"
// #include "examples/imgui_impl_opengl2.h"
#include "examples/imgui_impl_opengl3.h"

UI::UI(GLFWwindow* window) : platform{window} {
    ImGui::StyleColorsDark();
}

UI::~UI() {
}

UI::ImGuiContext::ImGuiContext() {
    ImGui::CreateContext();
}

UI::ImGuiContext::~ImGuiContext() {
    ImGui::DestroyContext();
}

UI::GlfwPlatform::GlfwPlatform(GLFWwindow* window) {
    ImGui_ImplGlfw_InitForOpenGL(window, true);
}

UI::GlfwPlatform::~GlfwPlatform() {
    ImGui_ImplGlfw_Shutdown();
}

UI::OpenGlInit::OpenGlInit() {
    ImGui_ImplOpenGL3_Init();
}

UI::OpenGlInit::~OpenGlInit() {
    ImGui_ImplOpenGL3_Shutdown();
}

void UI::render(const UIData& data) const {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    draw(data);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void UI::draw(const UIData& data) const {
    //     static bool show_demo_window = true;
    //     if (show_demo_window)
    //         ImGui::ShowDemoWindow(&show_demo_window);
    drawSceneWindow(data);
    drawMeshWindow(data);
    drawSampler(data);
}

void UI::drawSceneWindow(const UIData& uidata) const {
    if (!ImGui::Begin("Scene", nullptr, ImGuiWindowFlags_MenuBar)) {
        ImGui::End();
        return;
    }
    for (const Scene::ItemData& data : uidata.scene->getItems()) {
        if (ImGui::CollapsingHeader(("Item " + data.renderable->getName()).c_str())) {
            bool shown = !data.renderable->isHidden();
            if (ImGui::Checkbox(("Show##" + data.renderable->getName()).c_str(), &shown))
                data.renderable->setHidden(!shown);
            std::vector<FloatOption> options = data.renderable->getOptions();
            for (FloatOption& o : options) {
                ImGui::InputFloat((o.getName() + "##" + data.renderable->getName()).c_str(),
                                  &o.get(), 0.01, 0.1);
            }
            std::vector<BoolOption> boolOptions = data.renderable->getBoolOptions();
            for (BoolOption& o : boolOptions) {
                ImGui::Checkbox((o.getName() + "##" + data.renderable->getName()).c_str(),
                                &o.get());
            }
        }
    }
    ImGui::End();
}

static void drawMeshDirs(const FileManager::Directory& dir, const std::string& prefix = "") {
    auto removePrefix = [&prefix](std::string s) {
        if (s.size() >= prefix.length() && s.substr(0, prefix.length()) == prefix)
            return s.substr(prefix.length());
        return s;
    };
    const std::string folderName = removePrefix(dir.path.string());
    if (ImGui::TreeNode(("Folder " + folderName).c_str())) {
        for (const FileManager::Directory& subDir : dir.subDirs) {
            drawMeshDirs(subDir, dir.path.string());
        }
        for (const auto& path : dir.files) {
            if (ImGui::Button(removePrefix(path.string()).c_str())) {
                try {
                    MeshManager::getSingleton()->load(path.string());
                }
                catch (const std::runtime_error& e) {
                    std::cerr << e.what() << std::endl;
                }
            }
        }
        ImGui::TreePop();
    }
}

void UI::drawMeshWindow(const UIData& data) const {
    if (!ImGui::Begin("Mesh", nullptr, ImGuiWindowFlags_MenuBar)) {
        ImGui::End();
        return;
    }
    MeshManager* mm = MeshManager::getSingleton();
    MeshManager::ModelData models = mm->getCurrentModel();
    ImGui::BeginGroup();
    ImGui::InputInt("Preproc rep", &SphereMesh::sPreprocessCount_option, 10, 100);
    ImGui::SliderFloat("Alpha", &SphereMesh::sMatrixAlpha_option, 0.0f, 1.0f);
    ImGui::Separator();
    ImGui::InputInt("Max rep", &SphereMesh::sProcessMax_option, 100, 1000);
    ImGui::SliderFloat("Bias", &SphereMesh::sShoothenBias_option, 0.0f, 1.0f);
    if (models.sphereMesh != nullptr) {
        if (models.sphereMesh->getStatus() == SphereMesh::Status::UPLOADED) {
            if (ImGui::Button("Rebuild")) {
                models.sphereMesh->rebuild();
            }
            ImGui::SameLine();
            if (ImGui::Button("Refine")) {
                models.sphereMesh->refine();
            }
            if (ImGui::Button("Save cache")) {
                try {
                    MeshManager::getSingleton()->save();
                    std::cout << "Success" << std::endl;  // TODO ui message
                }
                catch (const std::exception& e) {
                    // TODO ui message
                    std::cerr << "Expection in MeshManager::save(): " << e.what() << std::endl;
                }
            }
        }
        else {
            float progress = static_cast<float>(models.sphereMesh->getProgress())
                             / SphereMesh::sProcessMax_option * 100;
            if (ImGui::Button(("Terminate (" + std::to_string(progress) + "%)").c_str())) {
                models.sphereMesh->terminateBuild();
            }
        }
    }
    ImGui::EndGroup();
    ImGui::Separator();
    // if (ImGui::Button("Cube")) {
    //     MeshManager::getSingleton()->load(loadMeshCube());
    // }
    drawMeshDirs(FileManager::getSingleton()->getRoot());
    ImGui::End();
}

void UI::drawSampler(const UIData& data) const {
    if (!ImGui::Begin("Sampler", nullptr, ImGuiWindowFlags_MenuBar)) {
        ImGui::End();
        return;
    }
    MeshManager* mm = MeshManager::getSingleton();
    MeshManager::ModelData models = mm->getCurrentModel();
    if (models.sphereMesh == nullptr
        || models.sphereMesh->getStatus() != SphereMesh::Status::UPLOADED) {
        ImGui::End();
        return;
    }
    if (models.sampler != nullptr && models.sphereMesh != nullptr) {
        static int sphericalCoordinates = 128;
        static bool sphericalCoordinates_enabled = true;
        ImGui::Checkbox("Spherical coordinates##checkbox", &sphericalCoordinates_enabled);
        ImGui::InputInt("##SphericalNumber", &sphericalCoordinates);
        static int cubeMap = 128;
        static bool cubeMap_enabled = false;
        // ImGui::Checkbox("Cube map##checkbox", &cubeMap_enabled);
        // ImGui::InputInt("##CubeNumber", &cubeMap);
        static int tetrahedron = 128;
        static bool tetrahedron_enabled = false;
        // ImGui::Checkbox("Tetrahedron##checkbox", &tetrahedron_enabled);
        // ImGui::InputInt("##TetrahedronNumber", &tetrahedron);
        static int octahedron = 128;
        static bool octahedron_enabled = true;
        ImGui::Checkbox("Octahedron##checkbox", &octahedron_enabled);
        ImGui::InputInt("##OctahedronNumber", &octahedron);
        if (ImGui::Button("Build##sampler")) {
            Sampler::BuildData buildData{
              static_cast<unsigned int>(sphericalCoordinates_enabled ? sphericalCoordinates : 0),
              static_cast<unsigned int>(cubeMap_enabled ? cubeMap : 0),
              static_cast<unsigned int>(tetrahedron_enabled ? tetrahedron : 0),
              static_cast<unsigned int>(octahedron_enabled ? octahedron : 0)};
            models.sampler->build(buildData, models.sphereMesh);
        }
        ImGui::Separator();
        Sampler::BuildData buildData = models.sampler->getBuildData();
        auto sameLine = [used = false]() mutable {
            // if (used)
            //     ImGui::SameLine();
            // used = true;
        };
        if (models.sampler->isBuilt()) {
            if (buildData.sphericalCoordinates > 0) {
                sameLine();
                if (ImGui::Button("Spherical coordinates##buildButton"))
                    models.sampler->display(Sampler::MapMode::SPHERICAL_COORDINATES);
            }
            // if (buildData.cubeMap > 0) {
            //     sameLine();
            //     if (ImGui::Button("Cube map##buildButton"))
            //         models.sampler->display(Sampler::MapMode::CUBE_MAP);
            // }
            // if (buildData.tetrahedron > 0) {
            //     sameLine();
            //     if (ImGui::Button("Tetrahedron##buildButton"))
            //         models.sampler->display(Sampler::MapMode::TETRAHEDRON);
            // }
            if (buildData.octahedron > 0) {
                sameLine();
                if (ImGui::Button("Octahedron##buildButton"))
                    models.sampler->display(Sampler::MapMode::OCTAHEDRON);
            }
        }
        ImGui::Separator();
    }
    ImGui::End();
}
