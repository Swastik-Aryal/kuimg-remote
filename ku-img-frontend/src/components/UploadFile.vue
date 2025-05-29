<template>
  <!-- Main container with background color -->
  <div id="app" style="background-color: #fcfaff;">
    <!-- Centered container for the file upload component -->
    <div style="padding: 30px; display: flex; justify-content: center;">
      <!-- Card-like container with border and padding -->
      <div style="border-radius: 10px; width: full; border: 1px solid #eee; background-color: white; padding: 15px;">
        
        <!-- Title of the file uploader -->
        <div style="margin-top: 10px; color:#c73c8be2; font-size:x-large; font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;">
          {{ title }}
        </div>

        <!-- Image preview if an image file is uploaded -->
        <div v-if="previewImage" class="imagePreviewWrapper" :style="{ 'background-image': `url(${previewImage})` }"></div>

        <!-- PDF preview if a PDF file is uploaded -->
        <div v-else-if="file && fileType === 'application/pdf'" class="filePreview">
          <span class="fa fa-file-pdf-o" style="font-size: 72px; color: #ed4c4c; margin-bottom: 10px;"></span>
          <p>{{ file.name }}</p>
        </div>

        <!-- Drop zone for file upload -->
        <div v-if="!file" style="display: flex; justify-content: center; width: full;">
          <div :class="['dropZone', dragging ? 'dropZone-over' : '']" @dragenter="dragging = true" @dragleave="dragging = false">
            <div class="dropZone-info" @drag="onChange">
              <span class="fa fa-cloud-upload dropZone-title" style="margin-right: 8px;"></span>
              <span class="dropZone-title">Drop file or click to upload</span>
              <div class="dropZone-upload-limit-info">
                <br>
                <div>extension support: {{ acceptedExtensionsText }}</div>
                <div>maximum file size: {{ maxFileSizeMB }} MB</div>
              </div>
            </div>
            <!-- Hidden file input for manual file selection -->
            <input type="file" :accept="fileType" @change="onChange">
          </div>
        </div>

        <!-- Uploaded file actions (submit or remove) -->
        <div v-else style="display: flex; justify-content: center; width: full;">
          <div class="dropZone-uploaded" style="margin: 10px; display:flex; flex-direction: row; align-items: center;">
            <div class="dropZone-uploaded-info">
              <button type="button" class="btn btn-primary removeFile" @click="submitFile">Submit</button>
              <button type="button" class="btn btn-danger removeFile" @click="removeFile">Remove File</button>
            </div>
          </div>
        </div>

        <!-- Display uploaded file information -->
        <div v-if="file" class="uploadedFile-info" style="margin-bottom: 10%; text-align: center;">
          <div>Filename: {{ file.name }}</div>
          <div>Filesize(bytes): {{ file.size }}</div>
          <div>extension：{{ extension }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'FileUpload',
  props: {
    // Accepted file types (e.g., 'image/*', 'application/pdf')
    fileType: {
      type: String,
      required: true,
    },
    // Maximum file size in MB
    maxFileSizeMB: {
      type: Number,
      default: 5,
    },
    // Title of the uploader
    title: {
      type: String,
      default: 'File Uploader',
    },
  },
  data() {
    return {
      file: '', // Uploaded file object
      dragging: false, // Dragging state for drop zone
      previewImage: null, // Preview image URL for image files
    };
  },
  computed: {
    // Compute accepted file extensions for display
    acceptedExtensions() {
      if (!this.fileType) {
        return [];
      }
      return this.fileType
        .split(',')
        .map((type) => {
          if (type === 'image/*') {
            return 'jpg, png, jpeg';
          }
          if (type === 'application/pdf') {
            return 'pdf';
          }
          const parts = type.split('/');
          if (parts.length === 2) {
            return parts[1];
          }
          if (type.startsWith('.')) {
            return type.substring(1);
          }
          return type;
        })
        .join(', ');
    },
    // Text representation of accepted extensions
    acceptedExtensionsText() {
      return this.acceptedExtensions;
    },
    // Extract file extension from uploaded file
    extension() {
      return this.file ? this.file.name.split('.').pop() : '';
    },
    // Maximum file size in bytes
    maxFileSize() {
      return this.maxFileSizeMB * 1024 * 1024;
    },
  },
  methods: {
    // Handle file input change or drop
    onChange(e) {
      var files = e.target.files || e.dataTransfer.files;

      if (!files.length) {
        this.dragging = false;
        return;
      }
      this.createFile(files[0]);
    },
    // Validate and set the uploaded file
    createFile(file) {
      const allowedTypes = this.fileType.split(',').map((type) => type.trim());
      const isFileTypeAllowed = allowedTypes.some((allowedType) => {
        if (allowedType === 'image/*') {
          return file.type.startsWith('image/');
        }
        return file.type === allowedType;
      });

      if (!isFileTypeAllowed) {
        alert(`Invalid file type. Please upload a file with one of the following types: ${this.acceptedExtensionsText}`);
        this.dragging = false;
        return;
      }

      if (file.size > this.maxFileSize) {
        alert(`Please check file size. It should not exceed ${this.maxFileSizeMB} MB.`);
        this.dragging = false;
        return;
      }
      this.file = file;
      this.dragging = false;

      console.log(this.file);

      // Generate preview for image files
      if (this.file.type.startsWith('image/')) {
        this.previewImage = URL.createObjectURL(this.file);
      } else {
        this.previewImage = null;
      }
    },
    // Emit the uploaded file to the parent component
    submitFile() {
      this.$emit('newFile', this.file);
    },
    // Remove the uploaded file
    removeFile() {
      this.file = '';
      this.previewImage = null;
    },
  },
};
</script>

<style scoped>
/* Styles for the drop zone */
.dropZone {
  border: 2px dashed #ccc;
  border-radius: 10px;
  padding: 2px;
  text-align: center;
  cursor: pointer;
  width: 300px; /* Adjust width as needed */
}

/* Highlight drop zone when dragging */
.dropZone-over {
  border-color: #c73c8be2;
  background-color: #f0f0f9;
}

/* Drop zone informational text */
.dropZone-info {
  color: #999;
}

/* Drop zone title styling */
.dropZone-title {
  font-size: 1.2em;
}

/* Drop zone upload limit information */
.dropZone-upload-limit-info {
  font-size: 0.9em;
  color: #666;
}

/* Hidden file input styling */
.dropZone input[type='file'] {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  cursor: pointer;
}

/* Uploaded file container styling */
.dropZone-uploaded {
  text-align: center;
}

/* Image preview container styling */
.imagePreviewWrapper {
  width: 200px; /* Adjust as needed */
  height: 200px; /* Adjust as needed */
  margin: 10px auto;
  background-size: contain;
  background-repeat: no-repeat;
  background-position: center;
  border: 1px solid #eee;
  border-radius: 5px;
}

/* PDF file preview styling */
.filePreview {
  text-align: center;
  margin: 20px auto;
}

/* Uploaded file information styling */
.uploadedFile-info {
  text-align: center;
  margin-top: 40px;
  color: #555;
}

/* Remove file button styling */
.removeFile {
  margin: 10px;
}
</style>