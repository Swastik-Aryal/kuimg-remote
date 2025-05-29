# image-tagging

## Project setup
```
npm install
```

### Compiles and hot-reloads for development
```
npm run serve
```

### Compiles and minifies for production
```
npm run build
```

### Customize configuration
See [Configuration Reference](https://cli.vuejs.org/config/).



# PROJECT OVER VIEW for Page 3 component:

# Component Documentation: Page3View.vue

## 📘 Overview

This Vue component handles the workflow for **image classification model training, testing, and saving**. It enables users to:
* Upload an image.
* Automatically or manually tag it.
* Trigger model training.
* Evaluate the model.
* Save or retrain based on accuracy.

## 🧩 Key Functional Features

### 1. Image Upload & Auto-tagging
* 📥 Component: `<UploadFile />`
* 🔁 Trigger: `onFileChange(file)`
* 📡 Endpoint: `POST /tag`
* 📦 Payload: FormData with key `img`
* 📤 Response:
```json
{
  "tags": ["tag1", "tag2"] 
  // OR 
  "tags": "We dont have model to recognise this, please train it first"
}
```
* 🧠 If the model is not trained for this image, the user is asked to input a custom label to begin training.

### 2. Custom Label Input & Training
* 📥 Input: `customString`
* 🚀 Trigger: `submitCustomString()`
* 📡 Endpoint: `POST /autotag/img/fetch_train`
* 📦 Payload:
```json
{
  "tag": "custom_label"
}
```
* 🧪 Model Accuracy Logic:
   * > 0.99: ✅ Model is saved via `/autotag/model/register`.
   * 0.20 - 0.98: ⚠️ Requires manual test confirmation (`/autotag/tagTestVerify`).
   * < 0.20: ❌ Retry up to 2 times. If still low, show "Run GAN and Train" option.

### 3. Model Save
* 💾 Trigger: `manual_confirm_save()`
* 📡 Endpoint: `POST /autotag/model/register`
* 📦 Payload:
```json
{
  "template": "keras/MultiClassSingleTagKerasStandardModelTemplateA.py",
  "group": "custom_label",
  "model_key": "custom_label_model.zip"
}
```
* ✅ Registers the model with backend for future predictions.

### 4. Test Image Upload & Validation
* 📥 Component: `<TestUploadFile />`
* 🔁 Trigger: `onTestFileChange(file)`
* 📡 Endpoint: `POST /autotag/tagTestVerify`
* 📦 Payload: FormData with keys:
   * `img`: Test image
   * `temp_tag`: User-entered label
* 📤 Response:
```json
{
  "tags": ["predicted_label"]
}
```

### 5. GAN + CNN Re-training (Fallback)
* 🚀 Trigger: `runGanAndTrain()`
* 📡 Endpoint: `POST /autotag/ml/train/fetch_gan_cnn`
* 📦 Payload:
```json
{
  "tag": "custom_label"
}
```
* 🧪 Logic:
   * > 0.99: Model saved and registered.
   * 0.20 - 0.98: Prompt user for test confirmation.
   * < 0.20: Retry twice, else fail.

## 🎯 UI States & Data Flow

| **State** | **Description** |
|-----------|-----------------|
| showTags | Display recognized tags from `/tag` |
| showStringField | Show label input if model not trained |
| showTrainButton | Show "Run GAN and Train" button if training fails |
| testFile | Prompt user for test image if accuracy < 99% |
| showTestTags | Show predicted tags from test image |
| modelSaveSuccess | Show success message if model is saved |
| modelSaveError | Show error and retry options if model training fails |

## 🧠 Retry Logic
* Both `submitCustomString` and `runGanAndTrain` retry **up to 2 times** if the validation accuracy is too low (< 0.20).
* Delays of 1 second are added between retries to avoid spamming the backend.

## ✅ Summary of Backend Endpoints

| **Endpoint** | **Method** | **Purpose** |
|--------------|------------|-------------|
| `/tag` | POST | Recognize image tags |
| `/autotag/img/fetch_train` | POST | Train model with custom label |
| `/autotag/model/register` | POST | Save trained model |
| `/autotag/tagTestVerify` | POST | Validate model with test image |
| `/autotag/ml/train/fetch_gan_cnn` | POST | GAN + CNN fallback training |