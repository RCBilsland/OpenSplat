#include "cv_utils.hpp"
#include <cmath>

// Helper function to apply fisheye distortion model
cv::Point2f distortPointFisheye(const cv::Point2f& normalizedPt, float k1, float k2, float k3, float k4) {
    double x = normalizedPt.x;
    double y = normalizedPt.y;
    double r = std::sqrt(x*x + y*y);
    
    if (r > 0) {
        double theta = std::atan(r);
        double theta2 = theta * theta;
        double theta4 = theta2 * theta2;
        double theta6 = theta4 * theta2;
        double theta8 = theta4 * theta4;
        
        double theta_d = theta * (1 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8);
        double scale = theta_d / r;
        
        return cv::Point2f(x * scale, y * scale);
    }
    return normalizedPt;
}

// Helper function to remove fisheye distortion using iterative method
cv::Point2f undistortPointFisheye(const cv::Point2f& distortedPt, float k1, float k2, float k3, float k4) {
    cv::Point2f guess = distortedPt; // Initial guess
    const int maxIterations = 20;
    const float epsilon = 1e-7f;
    
    for (int i = 0; i < maxIterations; i++) {
        cv::Point2f distorted = distortPointFisheye(guess, k1, k2, k3, k4);
        cv::Point2f error(distortedPt.x - distorted.x, distortedPt.y - distorted.y);
        
        if (error.x*error.x + error.y*error.y < epsilon)
            break;
            
        guess.x += error.x;
        guess.y += error.y;
    }
    
    return guess;
}

cv::Point2f projectFisheye(const cv::Point3f &p3d, const cv::Mat &K, const cv::Mat &D) {
    // Project using fisheye model with complete distortion coefficients
    double x = p3d.x;
    double y = p3d.y;
    double z = p3d.z;
    
    // Convert to normalized coordinates
    if (std::abs(z) < 1e-7) z = 1e-7;
    cv::Point2f normalizedPt(x/z, y/z);
    
    // Apply fisheye distortion
    cv::Point2f distortedPt = distortPointFisheye(normalizedPt, 
                                                 D.at<double>(0), D.at<double>(1),
                                                 D.at<double>(2), D.at<double>(3));
    
    // Apply intrinsic parameters
    cv::Point2f p2d;
    p2d.x = K.at<double>(0,0) * distortedPt.x + K.at<double>(0,2);
    p2d.y = K.at<double>(1,1) * distortedPt.y + K.at<double>(1,2);
    
    return p2d;
}

cv::Mat imreadRGB(const std::string &filename){
    cv::Mat cImg = cv::imread(filename);

    if (cImg.empty()){
        std::cerr << "Cannot read " << filename << std::endl
                  << "Make sure the path to your images is correct" << std::endl;
        exit(1);
    }

    cv::cvtColor(cImg, cImg, cv::COLOR_BGR2RGB);
    return cImg;
}

void imwriteRGB(const std::string &filename, const cv::Mat &image){
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_RGB2BGR);
    cv::imwrite(filename, rgb);
}

cv::Mat floatNxNtensorToMat(const torch::Tensor &t){
    return cv::Mat(t.size(0), t.size(1), CV_32F, t.data_ptr());
}

torch::Tensor floatNxNMatToTensor(const cv::Mat &m){
    return torch::from_blob(m.data, { m.rows, m.cols }, torch::kFloat32).clone();
}

cv::Mat tensorToImage(const torch::Tensor &t){
    try {
        // Initial tensor validation
        if (!t.defined()) {
            throw std::runtime_error("Input tensor is undefined");
        }
        if (t.dim() != 3) {
            throw std::runtime_error("Image tensor must be 3-dimensional (height x width x channels), got " + 
                                   std::to_string(t.dim()) + " dimensions");
        }
        if (t.numel() == 0) {
            throw std::runtime_error("Input tensor has zero elements");
        }
        
        int h = t.size(0);
        int w = t.size(1);
        int c = t.size(2);

        if (c != 3) {
            throw std::runtime_error("Image tensor must have 3 channels (RGB), got " + 
                                   std::to_string(c) + " channels");
        }
        
        std::cout << "Converting tensor " << h << "x" << w << "x" << c << " to image" << std::endl;
        
        // Create image and validate
        cv::Mat image(h, w, CV_8UC3);
        if (image.empty()) {
            throw std::runtime_error("Failed to create output image");
        }
        
        // Scale, clamp, and convert tensor
        torch::Tensor scaledTensor = (t * 255.0).clamp(0, 255).to(torch::kU8).contiguous();
        if (!scaledTensor.defined() || scaledTensor.numel() == 0) {
            throw std::runtime_error("Failed to scale and convert tensor");
        }
        
        // Verify memory sizes before copy
        size_t tensorSize = scaledTensor.numel() * scaledTensor.element_size();
        size_t imageSize = image.total() * image.elemSize();
        if (tensorSize != imageSize) {
            throw std::runtime_error("Memory size mismatch: Tensor=" + 
                                   std::to_string(tensorSize) + " bytes, Image=" + 
                                   std::to_string(imageSize) + " bytes");
        }
        
        // Copy data
        std::memcpy(image.data, scaledTensor.data_ptr(), imageSize);
        
        std::cout << "Image conversion successful" << std::endl;
        return image;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("tensorToImage failed: " + std::string(e.what()));
    }
}

torch::Tensor imageToTensor(const cv::Mat &image){
    try {
        // Initial validation
        if (image.empty()) {
            throw std::runtime_error("Input image is empty");
        }
        if (image.channels() != 3) {
            throw std::runtime_error("Image must have 3 channels (RGB), got " + 
                                   std::to_string(image.channels()));
        }
        if (image.type() != CV_8UC3) {
            throw std::runtime_error("Image must be 8-bit RGB, got type " + 
                                   std::to_string(image.type()));
        }
        
        // Make continuous copy if needed
        cv::Mat continuous;
        if (!image.isContinuous()) {
            std::cout << "Making continuous copy of image" << std::endl;
            image.copyTo(continuous);
        } else {
            continuous = image;
        }
        
        // Verify continuous copy
        if (!continuous.isContinuous()) {
            throw std::runtime_error("Failed to create continuous image");
        }
        
        std::cout << "Creating tensor from image: " 
                  << continuous.rows << "x" << continuous.cols 
                  << "x" << continuous.channels() << std::endl;
        
        // Create contiguous CPU tensor
        auto options = torch::TensorOptions()
            .dtype(torch::kU8)
            .layout(torch::kStrided)
            .requires_grad(false)
            .device(torch::kCPU);
        
        // Create and verify shape
        auto shape = std::vector<int64_t>{continuous.rows, continuous.cols, 3};
        auto img = torch::empty(shape, options);
        
        // Verify tensor creation
        if (!img.defined()) {
            throw std::runtime_error("Failed to create empty tensor");
        }
        
        // Copy data with size verification
        size_t dataSize = continuous.total() * continuous.elemSize();
        size_t tensorSize = img.numel() * img.element_size();
        
        if (dataSize != tensorSize) {
            throw std::runtime_error("Memory size mismatch: Image=" + 
                                   std::to_string(dataSize) + " bytes, Tensor=" + 
                                   std::to_string(tensorSize) + " bytes");
        }
        
        std::memcpy(img.data_ptr(), continuous.data, dataSize);
        
        // Convert to float and normalize
        std::cout << "Converting to float32 and normalizing" << std::endl;
        auto result = img.to(torch::kFloat32).div(255.0);
        
        // Final validation with detailed dimensions
        if (!result.defined()) {
            throw std::runtime_error("Final tensor is undefined");
        }
        if (result.numel() == 0) {
            throw std::runtime_error("Final tensor has zero elements");
        }
        if (result.dim() != 3) {
            throw std::runtime_error("Final tensor has wrong number of dimensions: " + 
                                   std::to_string(result.dim()));
        }
        
        std::cout << "Final tensor created successfully: ["
                  << result.size(0) << ", "
                  << result.size(1) << ", "
                  << result.size(2) << "]" << std::endl;
        
        return result;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("imageToTensor failed: " + std::string(e.what()));
    }
}

