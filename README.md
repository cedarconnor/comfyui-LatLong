# ComfyUI LatLong - Equirectangular Image Processing Nodes

Advanced equirectangular (360°) image processing nodes for ComfyUI, enabling precise rotation, horizon adjustment, and specialized cropping operations for panoramic images.

## 🌟 Features

### 🔄 **3D Rotation Control**
- **Yaw Rotation**: -180° to +180° (horizontal rotation)
- **Pitch Rotation**: -90° to +90° (vertical tilt, defaults to -65°)
- **Roll Rotation**: -180° to +180° (banking/roll)
- **Horizon Offset**: -90° to +90° (vertical horizon adjustment)

### ✂️ **Specialized Cropping**
- **180° Longitude Crop**: Extract center 180° from full 360° panorama
- **Square Crop**: Crop width to match original image height (perfect for square outputs)
- **Custom Dimensions**: Configurable output width and height for 180° crops

### 🎨 **High-Quality Filtering**
- **Lanczos** (default): Highest quality, best for photographic content
- **Bicubic**: High quality, good balance of speed and quality
- **Bilinear**: Medium quality, faster processing
- **Nearest**: Lowest quality, fastest processing

### ⚡ **Performance Features**
- **Batch Processing**: Handle multiple images simultaneously
- **Progress Indicators**: Real-time processing feedback
- **Float32 Pipeline**: Maintains image quality throughout processing
- **Memory Efficient**: Optimized for large panoramic images

## 📦 Nodes Included

### 1. **Equirectangular Rotate**
Dedicated node for rotation and horizon adjustments only.
- All 3D rotation controls (yaw, pitch, roll)
- Horizon offset adjustment
- Interpolation quality selection

### 2. **Equirectangular Crop 180**
Specialized node for 180-degree longitude cropping.
- Center 180° extraction from 360° panoramas
- Custom output dimensions
- Aspect ratio maintenance option
- High-quality resize with selectable interpolation

### 3. **Equirectangular Crop Square**
Creates perfect square crops from equirectangular images.
- Width automatically set to original image height
- Ideal for social media and square displays
- Center-aligned cropping

### 4. **Equirectangular Processor (All-in-One)**
Complete processing node combining all features.
- All rotation controls
- Both cropping options (square takes precedence)
- Full interpolation control
- Single-node workflow solution

## 🚀 Installation

### Method 1: Clone Repository
1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/cedarconnor/comfyui-LatLong.git
   ```

3. Install dependencies:
   ```bash
   cd comfyui-LatLong
   pip install -r requirements.txt
   ```

4. Restart ComfyUI

### Method 2: Manual Installation
1. Download and extract this repository to `ComfyUI/custom_nodes/comfyui-LatLong/`
2. Install the required dependencies:
   ```bash
   pip install numpy>=1.21.0 opencv-python>=4.5.0 scipy>=1.7.0 torch>=1.9.0
   ```
3. Restart ComfyUI

## 📋 Requirements

- **ComfyUI**: Latest version
- **Python**: 3.8+
- **Dependencies**:
  - `numpy>=1.21.0`
  - `opencv-python>=4.5.0` 
  - `scipy>=1.7.0`
  - `torch>=1.9.0`

## 🎯 Usage Examples

### Basic Rotation
1. Add **Equirectangular Rotate** node
2. Connect your equirectangular image input
3. Adjust rotation parameters:
   - Yaw: Rotate left/right
   - Pitch: Tilt up/down (defaults to -65°)
   - Roll: Banking rotation
   - Horizon Offset: Vertical horizon adjustment

### Square Crop for Social Media
1. Add **Equirectangular Crop Square** node
2. Connect your panoramic image
3. Get a perfect square crop at original height resolution

### 180° View Extraction
1. Add **Equirectangular Crop 180** node
2. Set desired output dimensions
3. Extract center 180° field of view with high-quality resizing

### Complete Processing Pipeline
1. Add **Equirectangular Processor** node
2. Configure rotation (pitch defaults to -65°)
3. Enable square crop or 180° crop as needed
4. Select interpolation quality (Lanczos recommended)

## ⚙️ Technical Details

### Coordinate System
- Uses standard spherical coordinate transformations
- Proper handling of equirectangular projection mathematics
- Robust pole singularity handling

### Interpolation Methods
- **Lanczos**: Uses OpenCV LANCZOS4 and scipy spline interpolation
- **Bicubic**: Third-order polynomial interpolation
- **Bilinear**: Linear interpolation between pixels
- **Nearest**: Closest pixel sampling

### Image Processing Pipeline
1. **Input**: ComfyUI tensor format (B, H, W, C)
2. **Conversion**: Float32 processing to maintain quality
3. **Transformation**: 3D rotation matrix applications
4. **Interpolation**: High-quality resampling
5. **Output**: Properly formatted ComfyUI tensors

## 🔧 Advanced Features

### Longitude Wrapping
Proper handling of longitude boundaries for seamless 360° rotations.

### Batch Processing
Efficiently process multiple images with progress tracking.

### Memory Optimization
Designed to handle large panoramic images without excessive memory usage.

## 🐛 Troubleshooting

### Common Issues
- **"NoneType validation error"**: Fixed in latest version - restart ComfyUI
- **Memory errors**: Reduce image size or process fewer images in batch
- **Blurry output**: Use Lanczos or Bicubic interpolation for better quality

### Performance Tips
- Use Lanczos for final output quality
- Use Bilinear for faster preview/testing
- Process smaller batches for large images

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source. Please check the license file for details.

## 🙏 Acknowledgments

- Built for the ComfyUI ecosystem
- Uses scipy for robust mathematical transformations
- OpenCV for high-quality image processing

---

**Perfect for**: 360° photography, VR content creation, panoramic image processing, social media content, architectural visualization, and any workflow requiring precise equirectangular image manipulation.