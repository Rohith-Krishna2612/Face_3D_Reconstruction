# 🎨 Web Interface Layout Explanation

## What You'll See on the Website

### Layout Overview

When you upload an image, the website will show:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Face Restoration Demo                            │
│                  Upload Your Face Image                              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         📸 Original Image                            │
│                                                                       │
│                     [Your Uploaded Image]                            │
│                        (Restored Version)                            │
└─────────────────────────────────────────────────────────────────────┘

         🔄 Degradation & Restoration Comparisons
   Each row shows: Original Restored → Degraded → Restored from Degraded

┌─────────────────────────────────────────────────────────────────────┐
│                    🌫️ Gaussian Blur                                  │
├──────────────────┬──────────────────┬──────────────────────────────┤
│  Original        │   Degraded       │   Restored from Degraded     │
│  Restored        │   (Blurred)      │   (De-blurred)               │
│  [Image 1]       │   [Image 2]      │   [Image 3]                  │
│                  │                  │                              │
├──────────────────┴──────────────────┴──────────────────────────────┤
│  🔵 Clean Input  →  🔴 Degraded  →  🟢 AI Restored     Click to zoom│
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    🔇 Gaussian Noise                                 │
├──────────────────┬──────────────────┬──────────────────────────────┤
│  Original        │   Degraded       │   Restored from Degraded     │
│  Restored        │   (Noisy)        │   (Denoised)                 │
│  [Image 1]       │   [Image 2]      │   [Image 3]                  │
│                  │                  │                              │
├──────────────────┴──────────────────┴──────────────────────────────┤
│  🔵 Clean Input  →  🔴 Degraded  →  🟢 AI Restored     Click to zoom│
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    📱 JPEG Compression                               │
├──────────────────┬──────────────────┬──────────────────────────────┤
│  Original        │   Degraded       │   Restored from Degraded     │
│  Restored        │   (Compressed)   │   (Decompressed)             │
│  [Image 1]       │   [Image 2]      │   [Image 3]                  │
│                  │                  │                              │
├──────────────────┴──────────────────┴──────────────────────────────┤
│  🔵 Clean Input  →  🔴 Degraded  →  🟢 AI Restored     Click to zoom│
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    🔍 Downsampling                                   │
├──────────────────┬──────────────────┬──────────────────────────────┤
│  Original        │   Degraded       │   Restored from Degraded     │
│  Restored        │   (Low-res)      │   (Upscaled)                 │
│  [Image 1]       │   [Image 2]      │   [Image 3]                  │
│                  │                  │                              │
├──────────────────┴──────────────────┴──────────────────────────────┤
│  🔵 Clean Input  →  🔴 Degraded  →  🟢 AI Restored     Click to zoom│
└─────────────────────────────────────────────────────────────────────┘

                    🔄 Process New Image
```

---

## Detailed Explanation

### Each Row Shows 3 Images Side-by-Side:

#### Image 1 (Left): **Original Restored**
- Your uploaded image after initial restoration
- Clean, baseline version
- Same for all 4 rows (reference point)
- Purpose: Shows what the model produces from clean input

#### Image 2 (Middle): **Degraded Version**
- Original image with degradation applied
- Different for each row:
  - Row 1: Blurred
  - Row 2: Noisy
  - Row 3: Compressed
  - Row 4: Downsampled
- Purpose: Shows what the model needs to fix

#### Image 3 (Right): **Restored from Degraded**
- The degraded image after AI restoration
- Shows model's ability to fix that specific degradation
- Purpose: Demonstrates model's robustness

---

## Why This Layout?

### Purpose:
This comparison shows **how well the model handles different types of image quality issues**.

### What You Learn:
1. **Blur Row**: Can the model sharpen blurred faces?
2. **Noise Row**: Can the model remove grainy noise?
3. **Compression Row**: Can the model fix JPEG artifacts?
4. **Downsample Row**: Can the model upscale low-resolution faces?

### Key Insight:
By comparing the **original restored** (left) with the **restored from degraded** (right), you can see:
- ✅ Good restoration: Right image looks similar to left image
- ⚠️ Poor restoration: Right image still looks degraded

---

## User Interaction

### Features:
- **Click any image**: Opens full-size view in modal
- **Hover effect**: Images zoom slightly on hover
- **Color coding**:
  - 🔵 Blue badge = Clean input
  - 🔴 Red badge = Degraded
  - 🟢 Green badge = AI Restored
- **Responsive**: Works on desktop and mobile

### Processing Flow:
```
User Upload → Backend Processing → Results Display
    ↓
1. Apply restoration to clean image (original restored)
2. Create 4 degraded versions (blur, noise, compress, downsample)
3. Restore each degraded version
4. Display all in comparison grid
```

---

## Technical Details

### What the Backend Does:

```python
# 1. Get uploaded image
uploaded_image = user_input

# 2. Restore original (baseline)
original_restored = model.restore(uploaded_image, w=0.5)

# 3. For each degradation type:
for degradation_type in ['blur', 'noise', 'compress', 'downsample']:
    # Apply degradation
    degraded = apply_degradation(uploaded_image, degradation_type)
    
    # Restore degraded version
    restored_from_degraded = model.restore(degraded, w=0.5)
    
    # Send to frontend
    results[degradation_type] = {
        'degraded': degraded,
        'restored': restored_from_degraded
    }

# Also send original restored for comparison
results['original'] = original_restored
```

### Response Structure:
```json
{
  "original": "base64_image_string",
  "results": {
    "blur": {
      "degraded": "base64_blurred_image",
      "restored": "base64_restored_from_blur"
    },
    "gaussian_noise": {
      "degraded": "base64_noisy_image",
      "restored": "base64_restored_from_noise"
    },
    "jpeg_compression": {
      "degraded": "base64_compressed_image",
      "restored": "base64_restored_from_compression"
    },
    "downsampling": {
      "degraded": "base64_downsampled_image",
      "restored": "base64_restored_from_downsample"
    }
  },
  "processing_time": 2.34
}
```

---

## Example Use Case

### Scenario: Testing a Face Photo

1. **Upload**: Your selfie (512×512 or higher)
2. **See**: 
   - Original restored version (clean)
   - 4 degraded versions
   - 4 restored versions
3. **Compare**: Which degradations does the model handle best?
4. **Insight**: 
   - If blur restoration looks good → Model is great at sharpening
   - If noise restoration looks good → Model is great at denoising
   - If all look good → Model is robust to various degradations!

---

## Benefits of This Layout

### 1. **Comprehensive Testing**
Shows model performance on 4 different degradation types simultaneously.

### 2. **Clear Comparison**
Easy to see original → degraded → restored progression.

### 3. **Quality Assessment**
Quickly identify which degradations the model handles well/poorly.

### 4. **Educational**
Users understand what each degradation type means visually.

### 5. **Professional Presentation**
Clean, organized layout suitable for demos and presentations.

---

## Mobile View

On smaller screens, the 3 images stack vertically:

```
┌──────────────────┐
│  🌫️ Gaussian Blur │
├──────────────────┤
│  Original        │
│  Restored        │
│  [Image 1]       │
├──────────────────┤
│  Degraded        │
│  (Blurred)       │
│  [Image 2]       │
├──────────────────┤
│  Restored from   │
│  Degraded        │
│  [Image 3]       │
└──────────────────┘
```

---

## Customization Options

### Change Restoration Strength
In `backend/main.py`:
```python
restored, _ = model(degraded, w=0.5)  # w = 0.0 to 1.0
# w=0.0: Keep more original details
# w=0.5: Balanced (default)
# w=1.0: Maximum quality enhancement
```

### Change Degradation Intensity
In `config.yaml`:
```yaml
degradations:
  blur:
    kernel_size: [7, 9, 11]  # Increase for more blur
  gaussian_noise:
    sigma_range: [5, 50]     # Increase for more noise
  jpeg_compression:
    quality_range: [30, 95]  # Decrease min for more artifacts
  downsampling:
    scale_range: [0.25, 1.0] # Decrease min for lower resolution
```

---

## Summary

**What you get:**
- 1 original restored image (top)
- 4 comparison rows (one per degradation type)
- Each row: 3 images side-by-side
- Total: 13 images displayed per upload

**What it shows:**
- Model's baseline performance (original restored)
- Model's robustness to blur, noise, compression, and downsampling
- Side-by-side comparisons for easy quality assessment

**What you can do:**
- Click any image for full-size view
- Compare restorations visually
- Test with different face photos
- Share results with others

**Perfect for:**
- Demonstrating model capabilities
- Quality assessment
- Research presentations
- Portfolio showcases
- User-facing applications
