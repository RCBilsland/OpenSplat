#include <filesystem>
#include <nlohmann/json.hpp>
#include "input_data.hpp"
#include "cv_utils.hpp"

namespace fs = std::filesystem;
using namespace torch::indexing;
using json = nlohmann::json;

namespace ns{ InputData inputDataFromNerfStudio(const std::string &projectRoot); }
namespace cm{ InputData inputDataFromColmap(const std::string &projectRoot, const std::string& imageSourcePath); }
namespace osfm { InputData inputDataFromOpenSfM(const std::string &projectRoot); }
namespace omvg { InputData inputDataFromOpenMVG(const std::string &projectRoot); }

InputData inputDataFromX(const std::string &projectRoot, const std::string& colmapImageSourcePath){
    fs::path root(projectRoot);

    if (fs::exists(root / "transforms.json")){
        return ns::inputDataFromNerfStudio(projectRoot);
    }else if (fs::exists(root / "sparse") || fs::exists(root / "cameras.bin")){
        return cm::inputDataFromColmap(projectRoot, colmapImageSourcePath);
    }else if (fs::exists(root / "reconstruction.json")){
        return osfm::inputDataFromOpenSfM(projectRoot);
    }else if (fs::exists(root / "opensfm" / "reconstruction.json")){
        return osfm::inputDataFromOpenSfM((root / "opensfm").string());
    }else if (fs::exists(root / "sfm_data.json")){
        return omvg::inputDataFromOpenMVG((root).string());
    }
    else{
        throw std::runtime_error("Invalid project folder (must be either a colmap or nerfstudio or openmvg project folder)");
    }
}

torch::Tensor Camera::getIntrinsicsMatrix(){
    if (cameraType == CameraType::Fisheye) {
        // For fisheye, we'll use a simpler matrix since distortion is handled differently
        return torch::tensor({{fx, 0.0f, cx},
                            {0.0f, fy, cy},
                            {0.0f, 0.0f, 1.0f}}, torch::kFloat32);
    } else {
        return torch::tensor({{fx, 0.0f, cx},
                            {0.0f, fy, cy},
                            {0.0f, 0.0f, 1.0f}}, torch::kFloat32);
    }
}

void Camera::loadImage(float downscaleFactor){
    // Populates image and K, then updates the camera parameters
    // Caution: this function has destructive behaviors
    // and should be called only once
    if (image.numel()) throw std::runtime_error("loadImage already called for: " + filePath);
    
    if (filePath.empty()) {
        throw std::runtime_error("Camera has no file path set");
    }

    std::cout << "Starting image load for: " << filePath << std::endl;
    
    cv::Mat cImg = imreadRGB(filePath);
        
        // Verify image immediately after loading
        if (cImg.empty()) {
            throw std::runtime_error("Failed to load image (empty)");
        }
        if (cImg.dims != 2) {
            throw std::runtime_error("Invalid image dimensions: " + std::to_string(cImg.dims));
        }
        if (cImg.channels() != 3) {
            throw std::runtime_error("Invalid number of channels: " + std::to_string(cImg.channels()));
        }
        
        std::cout << "Initial image load successful - Size: " 
                  << cImg.cols << "x" << cImg.rows 
                  << ", Channels: " << cImg.channels()
                  << ", Type: " << cImg.type() << std::endl;
    // Additional validation and logging
    if (!cImg.isContinuous()) {
        cv::Mat temp;
        cImg.copyTo(temp);
        cImg = temp;
    }

    std::cout << "Successfully loaded image " << filePath 
              << " with dimensions " << cImg.cols << "x" << cImg.rows 
              << "x" << cImg.channels() << std::endl;
    
    float rescaleF = 1.0f;
    // If camera intrinsics don't match the image dimensions 
    if (cImg.rows != height || cImg.cols != width){
        rescaleF = static_cast<float>(cImg.rows) / static_cast<float>(height);
    }
    fx *= rescaleF;
    fy *= rescaleF;
    cx *= rescaleF;
    cy *= rescaleF;

    cv::Mat originalImg = cImg.clone(); // Keep a copy of original image
    
    if (downscaleFactor > 1.0f){
        float scaleFactor = 1.0f / downscaleFactor;
        cv::resize(cImg, cImg, cv::Size(), scaleFactor, scaleFactor, cv::INTER_AREA);
        
        if (cImg.empty() || cImg.cols == 0 || cImg.rows == 0) {
            throw std::runtime_error("Resize operation failed, resulting image is invalid");
        }
        
        std::cout << "Image resized from " << originalImg.cols << "x" << originalImg.rows 
                  << " to " << cImg.cols << "x" << cImg.rows << std::endl;
        
        fx *= scaleFactor;
        fy *= scaleFactor;
        cx *= scaleFactor;
        cy *= scaleFactor;
    }

    K = getIntrinsicsMatrix();
    cv::Rect roi;

    try {
        if (hasDistortionParameters()){
            cv::Mat cK = floatNxNtensorToMat(K);
            cv::Mat undistorted = cv::Mat::zeros(cImg.rows, cImg.cols, cImg.type());
            cv::Mat newK = cv::Mat::eye(3, 3, CV_32F);

            // Undistort based on camera type
            if (cameraType == CameraType::Fisheye) {
                std::vector<float> distCoeffs = undistortionParameters(true);
                cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
                    cK, distCoeffs, cv::Size(cImg.cols, cImg.rows), cv::Mat::eye(3, 3, CV_32F), 
                    newK, 1.0, cv::Size(cImg.cols, cImg.rows));
                cv::fisheye::undistortImage(cImg, undistorted, cK, distCoeffs, newK);
            } else {
                std::vector<float> distCoeffs = undistortionParameters(false);
                newK = cv::getOptimalNewCameraMatrix(cK, distCoeffs, cv::Size(cImg.cols, cImg.rows), 0, cv::Size(), &roi);
                cv::undistort(cImg, undistorted, cK, distCoeffs, newK);
            }

            // Verify undistorted image
            if (undistorted.empty() || undistorted.cols == 0 || undistorted.rows == 0) {
                throw std::runtime_error("Undistortion produced invalid image");
            }

            image = imageToTensor(undistorted);
            if (!image.defined() || image.numel() == 0) {
                throw std::runtime_error("Tensor creation from undistorted image failed");
            }

            K = floatNxNMatToTensor(newK);
        } else {
            roi = cv::Rect(0, 0, cImg.cols, cImg.rows);
            image = imageToTensor(cImg);
            if (!image.defined() || image.numel() == 0) {
                throw std::runtime_error("Tensor creation from image failed");
            }
        }

        // Validate before cropping
        if (!image.defined() || image.numel() == 0) {
            throw std::runtime_error("Invalid tensor before ROI cropping");
        }

        std::cout << "Pre-crop tensor dimensions: [" 
                  << image.size(0) << ", " << image.size(1) << ", " << image.size(2) 
                  << "], ROI: [" << roi.x << ", " << roi.y << ", " 
                  << roi.width << ", " << roi.height << "]" << std::endl;

        // Crop to ROI with validation
        if (roi.width > 0 && roi.height > 0) {
            auto cropped = image.index({
                Slice(roi.y, roi.y + roi.height),
                Slice(roi.x, roi.x + roi.width),
                Slice()
            });

            // Verify cropped tensor
            if (!cropped.defined() || cropped.numel() == 0) {
                throw std::runtime_error("ROI cropping produced invalid tensor");
            }

            image = cropped;
        }

        // Verify final tensor
        if (!image.defined() || image.numel() == 0) {
            throw std::runtime_error("Final tensor is invalid after ROI cropping");
        }

    } catch (const std::exception& e) {
        throw std::runtime_error("Image processing failed: " + std::string(e.what()));
    }

    try {
        // Update parameters
        height = image.size(0);
        width = image.size(1);
        fx = K[0][0].item<float>();
        fy = K[1][1].item<float>();
        cx = K[0][2].item<float>();
        cy = K[1][2].item<float>();

        // Comprehensive tensor validation
        if (!image.defined()) {
            throw std::runtime_error("Tensor is undefined");
        }
        if (image.numel() == 0) {
            throw std::runtime_error("Tensor has zero elements");
        }
        if (image.dim() != 3) {
            throw std::runtime_error("Tensor dimension is " + std::to_string(image.dim()) + ", expected 3");
        }
        if (image.size(0) == 0 || image.size(1) == 0 || image.size(2) != 3) {
            throw std::runtime_error("Invalid tensor dimensions: [" + 
                std::to_string(image.size(0)) + ", " + 
                std::to_string(image.size(1)) + ", " + 
                std::to_string(image.size(2)) + "]");
        }

        // Verify tensor values
        auto minmax = image.aminmax();
        auto min_val = std::get<0>(minmax);
        auto max_val = std::get<1>(minmax);
        
        if (min_val.item<float>() < 0.0f || max_val.item<float>() > 1.0f) {
            throw std::runtime_error("Tensor values out of range [0,1]");
        }

        std::cout << "Final tensor validation for " << filePath << ":\n"
                  << "  Dimensions: [" << image.size(0) << ", " 
                  << image.size(1) << ", " << image.size(2) << "]\n"
                  << "  Elements: " << image.numel() << "\n"
                  << "  Value range: [" << min_val.item<float>() 
                  << ", " << max_val.item<float>() << "]" << std::endl;

    } catch (const std::exception& e) {
        throw std::runtime_error("Tensor validation failed: " + std::string(e.what()));
    }
}

torch::Tensor Camera::getImage(int downscaleFactor) {
    // First verify that we have a valid image
    if (image.numel() == 0 || image.size(0) == 0 || image.size(1) == 0) {
        throw std::runtime_error("Invalid image: image has not been loaded or has zero dimensions");
    }

    if (downscaleFactor <= 1) return image;
    
    // Check if we already have this scale in our pyramid
    if (imagePyramids.find(downscaleFactor) != imagePyramids.end()){
        return imagePyramids[downscaleFactor];
    }

    // Convert tensor to OpenCV image
    cv::Mat cImg = tensorToImage(image);
    
    // Verify the OpenCV image is valid
    if (cImg.empty() || cImg.cols == 0 || cImg.rows == 0) {
        throw std::runtime_error("Failed to convert tensor to OpenCV image");
    }
    
    // Calculate new dimensions and verify they are valid
    int newWidth = cImg.cols / downscaleFactor;
    int newHeight = cImg.rows / downscaleFactor;
    
    // Make sure the resulting image won't be too small
    if (newWidth < 32 || newHeight < 32) {
        std::cerr << "Warning: Downscale factor " << downscaleFactor 
                  << " would result in image size " << newWidth << "x" << newHeight 
                  << " which is too small. Using original image." << std::endl;
        return image;
    }

    try {
        // Rescale the image
        cv::Mat resized;
        cv::resize(cImg, resized, cv::Size(newWidth, newHeight), 0.0, 0.0, cv::INTER_AREA);
        
        if (resized.empty() || resized.cols == 0 || resized.rows == 0 || resized.channels() != 3) {
            throw std::runtime_error("Resize operation produced invalid image");
        }

        // Convert to tensor with validation
        torch::Tensor t;
        try {
            t = imageToTensor(resized);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Tensor conversion failed: ") + e.what());
        }

        // Validate tensor
        if (t.numel() == 0 || t.dim() != 3 || t.size(0) == 0 || t.size(1) == 0 || t.size(2) != 3) {
            throw std::runtime_error("Generated tensor has invalid dimensions");
        }

        // Store in pyramid and return
        imagePyramids[downscaleFactor] = t;
        return t;
    } catch (const std::exception& e) {
        throw std::runtime_error("Error processing image " + filePath + ": " + e.what());
    }
}

bool Camera::hasDistortionParameters(){
    if (cameraType == CameraType::Fisheye) {
        return k1 != 0.0f || k2 != 0.0f || k3 != 0.0f || k4 != 0.0f;
    } else {
        return k1 != 0.0f || k2 != 0.0f || k3 != 0.0f || p1 != 0.0f || p2 != 0.0f;
    }
}

std::vector<float> Camera::undistortionParameters(bool fisheye){
    if (fisheye) {
        // For fisheye, OpenCV expects k1, k2, k3, k4 in this order
        std::vector<float> p = { k1, k2, k3, k4 };
        return p;
    } else {
        // For perspective, OpenCV expects k1, k2, p1, p2, k3, k4, k5, k6
        std::vector<float> p = { k1, k2, p1, p2, k3, 0.0f, 0.0f, 0.0f };
        return p;
    }
}

std::tuple<std::vector<Camera>, Camera *> InputData::getCameras(bool validate, const std::string &valImage){
    if (!validate) return std::make_tuple(cameras, nullptr);
    else{
        size_t valIdx = -1;
        std::srand(42);

        if (valImage == "random"){
            valIdx = std::rand() % cameras.size();
        }else{
            for (size_t i = 0; i < cameras.size(); i++){
                if (fs::path(cameras[i].filePath).filename().string() == valImage){
                    valIdx = i;
                    break;
                }
            }
            if (valIdx == -1) throw std::runtime_error(valImage + " not in the list of cameras");
        }

        std::vector<Camera> cams;
        Camera *valCam = nullptr;

        for (size_t i = 0; i < cameras.size(); i++){
            if (i != valIdx) cams.push_back(cameras[i]);
            else valCam = &cameras[i];
        }

        return std::make_tuple(cams, valCam);
    }
}


void InputData::saveCameras(const std::string &filename, bool keepCrs){
    json j = json::array();
    
    for (size_t i = 0; i < cameras.size(); i++){
        Camera &cam = cameras[i];

        json camera = json::object();
        camera["id"] = i;
        camera["img_name"] = fs::path(cam.filePath).filename().string();
        camera["width"] = cam.width;
        camera["height"] = cam.height;
        camera["fx"] = cam.fx;
        camera["fy"] = cam.fy;

        torch::Tensor R = cam.camToWorld.index({Slice(None, 3), Slice(None, 3)});
        torch::Tensor T = cam.camToWorld.index({Slice(None, 3), Slice(3,4)}).squeeze();
        
        // Flip z and y
        R = torch::matmul(R, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f})));

        if (keepCrs) T = (T / scale) + translation;

        std::vector<float> position(3);
        std::vector<std::vector<float>> rotation(3, std::vector<float>(3));
        for (int i = 0; i < 3; i++) {
            position[i] = T[i].item<float>();
            for (int j = 0; j < 3; j++) {
                rotation[i][j] = R[i][j].item<float>();
            }
        }

        camera["position"] = position;
        camera["rotation"] = rotation;
        j.push_back(camera);
    }
    
    std::ofstream of(filename);
    of << j;
    of.close();

    std::cout << "Wrote " << filename << std::endl;
}