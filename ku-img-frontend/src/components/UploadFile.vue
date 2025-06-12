<template>
  <div class="upload-panel">
    <h3 class="upload-panel__title">{{ title }}</h3>
    <p v-if="error" class="upload-panel__error">{{ error }}</p>

    <!-- The file input is now here, outside the v-if, but hidden via its class. -->
    <!-- This ensures the ref is always available. -->
    <input 
      ref="fileInput"
      type="file" 
      :accept="fileType" 
      @change="onChange" 
      class="drop-zone__input"
    >

    <!-- State 1: Drop Zone -->
    <div 
      v-if="!file" 
      :class="['drop-zone', dragging ? 'drop-zone--over' : '']" 
      @dragenter.prevent="dragging = true" 
      @dragover.prevent="dragging = true" 
      @dragleave.prevent="dragging = false"
      @drop.prevent="onDrop"
      @click="$refs.fileInput.click()"
    >
      <div class="drop-zone__info">
        <span class="drop-zone__icon fa fa-cloud-upload"></span>
        <p class="drop-zone__text">Drop file here or click to upload</p>
        <p class="drop-zone__subtext">
          Supports: {{ acceptedExtensionsText }} | Max Size: {{ maxFileSizeMB }}MB
        </p>
      </div>
    </div>

    <!-- State 2: File Preview & Actions -->
    <div v-else class="file-preview-container">
      <!-- Image Preview -->
      <div v-if="previewImage" class="file-icon-preview">
        <div class="image-preview" :style="{ 'background-image': `url(${previewImage})` }"></div>
        <p class="file-name">{{ file.name }}</p>
        <p class="file-size">{{ (file.size / 1024 / 1024).toFixed(2) }} MB</p>
      </div>
      
      <!-- PDF Preview Icon -->
      <div v-else-if="file.type === 'application/pdf'" class="file-icon-preview">
        <span class="fa fa-file-pdf-o" style="font-size: 72px; color: #D33838;"></span>
        <p class="file-name">{{ file.name }}</p>
        <p class="file-size">{{ (file.size / 1024 / 1024).toFixed(2) }} MB</p>
      </div>

      <!-- ZIP Preview Icon -->
      <div v-else-if="file.type.includes('zip') || file.name.endsWith('.zip')" class="file-icon-preview">
        <span class="fa fa-file-zip-o" style="font-size: 72px; color: #E8A133;"></span>
        <p class="file-name">{{ file.name }}</p>
        <p class="file-size">{{ (file.size / 1024 / 1024).toFixed(2) }} MB</p>
      </div>
      
      <!-- Action Buttons -->
      <div class="button-group" style="margin-top: 20px;">
        <button type="button" class="btn btn--primary" @click="submitFile">Submit</button>
        <button type="button" class="btn btn--secondary" @click="removeFile">Remove File</button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'UploadFile',
  props: {
    fileType: { type: String, required: true },
    maxFileSizeMB: { type: Number, default: 5 },
    title: { type: String, default: 'File Uploader' },
  },
  data() {
    return {
      file: null,
      dragging: false,
      previewImage: null,
      error: '',
    };
  },
  computed: {
    acceptedExtensionsText() {
      if (!this.fileType) return '';
      return this.fileType
        .split(',')
        .map(type => {
          if (type.trim() === 'image/*') return 'JPG, PNG';
          if (type.trim() === 'application/pdf') return 'PDF';
          if (type.trim().includes('zip')) return 'ZIP';
          return type.split('/')[1]?.toUpperCase() || type;
        })
        .join(', ');
    },
    maxFileSizeBytes() {
      return this.maxFileSizeMB * 1024 * 1024;
    },
  },
  methods: {
    onDrop(e) {
      this.dragging = false;
      const files = e.dataTransfer.files;
      if (!files || files.length === 0) return;
      this.createFile(files[0]);
    },
    onChange(e) {
      const files = e.target.files;
      if (!files || files.length === 0) return;
      this.createFile(files[0]);
    },
    createFile(file) {
      this.error = ''; // Reset error on new file selection

      // Validate file size
      if (file.size > this.maxFileSizeBytes) {
        this.error = `File exceeds max size of ${this.maxFileSizeMB}MB.`;
        return;
      }

      // Validate file type
      const allowedTypes = this.fileType.split(',').map(t => t.trim());
      const isValidType = allowedTypes.some(type => {
        if (type === 'image/*') return file.type.startsWith('image/');
        if (type.includes('zip')) return file.name.endsWith('.zip') || file.type.includes('zip');
        return file.type === type;
      });

      if (!isValidType) {
        this.error = `Invalid file type. Please upload: ${this.acceptedExtensionsText}`;
        return;
      }
      
      this.file = file;

      // Create image preview
      if (this.file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
          this.previewImage = e.target.result;
        };
        reader.readAsDataURL(this.file);
      } else {
        this.previewImage = null;
      }
    },
    submitFile() {
      if (this.file) {
        this.$emit('newFile', this.file);
        // Do not clear the file here, parent component might need to show it's been processed.
        // Parent component can call a method on this component to clear if needed.
      }
    },
    removeFile() {
      this.file = null;
      this.previewImage = null;
      this.error = '';
      // Reset the hidden file input so the same file can be selected again
      // This will now work because the ref is always present.
      if (this.$refs.fileInput) {
        this.$refs.fileInput.value = '';
      }
    },
  },
};
</script>

<style scoped>
/* Main Panel */
.upload-panel {
  width: 100%;
  background-color: #ffffff;
  border-radius: 12px;
  padding: 25px;
  text-align: center;
  border: 1px solid #e9ecef;
}
.upload-panel__title {
  font-size: 20px;
  font-weight: 600;
  color: #343a40;
  margin: 0 0 15px 0;
}
.upload-panel__error {
  color: #dc3545;
  font-weight: 500;
  background-color: #fbebed;
  padding: 10px;
  border-radius: 8px;
  margin-bottom: 15px;
}

/* Drop Zone Styling */
.drop-zone {
  border: 2px dashed #ced4da;
  border-radius: 10px;
  padding: 40px 20px;
  cursor: pointer;
  transition: all 0.2s ease-in-out;
  position: relative;
}
.drop-zone:hover {
  border-color: #158be3;
  background-color: #f8f9fa;
}
.drop-zone--over {
  border-color: #158be3;
  background-color: #e6f2ff;
}
.drop-zone__info {
  color: #6c757d;
}
.drop-zone__icon {
  font-size: 40px;
  color: #158be3;
}
.drop-zone__text {
  font-size: 18px;
  font-weight: 500;
  margin: 10px 0 5px 0;
  color: #495057;
}
.drop-zone__subtext {
  font-size: 14px;
  margin: 0;
}
.drop-zone__input {
  display: none;
}

/* File Preview Styling */
.file-preview-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 15px;
}
.image-preview {
  width: 100%;
  max-width: 250px;
  height: 250px;
  background-size: cover;
  background-position: center;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}
.file-icon-preview {
  width:100%;
  max-width: 250px;
  text-align: center;
}
.file-icon-preview .fa {
  font-size: 72px;
}
.file-name {
  font-weight: 600;
  color: #343a40;
  margin-top: 10px;
  margin-bottom: 5px;
  word-break: break-all;
}
.file-size {
  font-size: 14px;
  color: #6c757d;
  margin: 0;
}

/* --- Buttons --- */
.btn {
  display: inline-block;
  border: none;
  padding: 12px 28px;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  text-align: center;
  text-decoration: none;
  transition: all 0.2s ease-in-out;
}
.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.btn--primary {
  background-color: #158be3;
  color: white;
}
.btn--primary:hover { background-color: #1172bb; }
.btn--primary:disabled {
  background-color: #a0cff2;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}
.btn--secondary {
  background-color: #6c757d;
  color: white;
}
.btn--secondary:hover { background-color: #5a6268; }
.btn--full-width {
  width: 100%;
  padding-top: 14px;
  padding-bottom: 14px;
}
.button-group {
  display: flex;
  justify-content: center;
  gap: 15px;
  margin-top: 10px;
}
</style>