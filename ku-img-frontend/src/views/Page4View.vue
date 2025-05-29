<template>
  <main style="background-color: #fcfaff; height:100vh;">
    <div>
    <!-- UploadFile component for uploading PDF files -->
    <UploadFile v-if="showUpload" fileType="application/pdf" title="PDF Uploader" :maxFileSizeMB="100" @newFile="onFileChange($event)" />
    </div>
    <!-- Loader displayed while extracting keywords -->
    <ClipLoader :loading="loading && !showKeywords && !trainingState" size="100px" msg="Extracting Keywords..." style="margin: 100px auto;"/>
    
    <!-- Display extracted keywords -->
    <div v-if="showKeywords" style="font-size: larger; width: 50%; max-width: fit-content; margin: 100px auto;">
    <p style="font-size: 30px; font-weight: bold; color:#d574af; text-align: center;">Extracted Keywords: </p>
    <ul style="padding: 0; margin: 0; list-style-position: inside;">
      <li v-for="(info, keyword) in keywords" :key="keyword" style="list-style-type: none; font-size: 20px; text-align: center; margin-top: 10px;">
      <label :for="`kw-${keyword}`" style="cursor: pointer; color: #6c757d;">
        <b>{{ keyword }}</b> (score: {{ info.score }})
      </label>
      <!-- Show model existence status -->
      <span v-if="info.exists" style="margin-left: 10px; color: #28a745;" >: Model exists.</span>
      <!-- Checkbox for selecting keywords to train -->
      <input v-else type="checkbox" :id="`kw-${keyword}`" :value="keyword" v-model="selectedKeywords" style="margin-left: 10px; transform: scale(1.2);"/>
      </li>
    </ul>
    <!-- Button to train selected keywords -->
    <button @click="trainKeywords" style="width: 150px; border: 2px solid #158be3; margin:30px; text-transform: none; border-radius: 5px; background-color: #158be3; box-shadow: 0 2px 4px white(0, 0, 0, 0.1);
      color: white; padding: 8px;">Train Selected</button>
    </div>  
  
    <!-- Training state UI -->
    <div v-if="trainingState" style=" display:flex; column-gap: 20px; margin: 100px auto;">
    <div style="flex: 0.5; white-space: nowrap; margin:20px auto">
      <p style="font-size: 30px; font-weight: bold; color:#d574af; text-align: center;">Keywords:</p>
      <ul style="padding: 0; margin: 0; list-style-position: inside;">
      <li v-for="(info, keyword) in keywords" :key="keyword" style="list-style-type: none; font-size: 20px; text-align: center; margin-top: 10px; white-space: nowrap;">
        <label :for="`kw-${keyword}`" style="cursor: pointer; color: #6c757d;">
        <b>{{ keyword }}</b>
        </label>
        <!-- Display training status -->
        <span v-if="info.exists" style="margin-left: 10px; color: #28a745;" >: Model already exists.</span>
        <span v-else-if="keyword === currentTraining" class="animated-fade">: Training...</span>
        <span v-else-if="trainedResults[keyword] !== undefined" style="margin-left: 10px; color: #28a745;" >: Model trained, accuracy: {{ trainedResults[keyword] }}%</span>
        <span v-else-if="!selectedKeywords.includes(keyword)" style="margin-left: 10px; color: #ffc107;">: Not selected.</span>
        <span v-else>: Queued...</span>
      </li>
      </ul>
    </div> 

    <!-- Training progress and manual confirmation -->
    <div style="flex: 1; border:#158be3 2px solid; border-radius: 20px; padding: 20px; background-color: white; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
      <p style="font-size: 30px; font-weight: bold; color:#d574af; text-align: center;">Training "{{ this.currentTraining }}" Model...</p>
      <ClipLoader :loading="loading" size="100px" :msg="loadingText" style="margin: 100px auto;"/>

      <!-- Error message if model save fails -->
      <div v-if="modelSaveError">
      <p style="font-size: medium; font-weight: bold; color: rgb(227, 73, 73); ">
        Model was not saved due to lower validation accuracy ie: {{ returnedValAccuracy * 100 }}%. Re-Autotraining in Progress.
      </p>
      <!-- Button to manually trigger GAN training -->
      <button v-if="showTrainButton" @click="runGAN = true" style="width: auto;
        border: 2px solid #158be3;
        text-transform: none;
        border-radius: 16px;
        background-color: #158be3;
        box-shadow: 0 2px 4px white(0, 0, 0, 0.1);
        color: white; padding: 10px;">{{ trainButtonText }}</button>
      </div>
    
      <!-- Manual testing UI -->
      <div v-if="testFile">
      <p style="font-size: medium; font-weight: bold; color: rgb(227, 73, 73); ">
        The model's accuracy was in between 50 to 85. 
        Thus we want you to test and confirm manually.
        Please upload a test file and confirm the model. 
      </p>
      <TestUploadFile @newFile="onTestFileChange($event)" />
      </div>
      <div v-if="showTestTags" >
      <div style="font-size: larger; color: rgb(112, 108, 108);">
        <p>The image is </p>
        {{ test_tags }}
      </div>
      <div>
        <p style="font-size: medium; font-weight: bold; ">
        Would you like to save the model or retrain it ?
      </p>
      <!-- Buttons for saving or retraining the model -->
      <button @click="manual_confirm_save" style="width: auto;
        border: 2px solid #158be3;
        text-transform: none;
        margin-right:20px;
        border-radius: 20px  20px  20px 20px;
        background-color: #158be3;
        box-shadow: 0 2px 4px white(0, 0, 0, 0.1);
        color: white; padding: 8px;">Save Model</button>
        <button @click="submitCustomString " style="width: auto;
        border: 2px solid #158be3;
        text-transform: none;
        border-radius: 20px  20px  20px 20px;
        background-color: #158be3;
        box-shadow: 0 2px 4px white(0, 0, 0, 0.1);
        color: white; padding: 8px;">Re-Train</button>
      </div>
      </div>
      <!-- Success message if model is saved -->
      <div v-if="modelSaveSuccess">
      Model was saved successfully!
      </div>
    </div>
    </div>
  </main>
</template>

<script>
  import UploadFile from '@/components/UploadFile.vue'
  import TestUploadFile from '@/components/UploadTestFile.vue'
  import ClipLoader from '@/components/ClipLoader.vue'
  import axios from 'axios'
  
  export default {
  name: 'Page4View',
  components: {
    UploadFile,
    TestUploadFile,
    ClipLoader,
  },
  data() {
    return {
    keywords: {}, // Stores extracted keywords
    selectedKeywords: [], // Stores selected keywords for training
    showUpload: true, // Controls visibility of the upload component
    customString: '', // Custom string for additional functionality
    showKeywords: false, // Controls visibility of the keywords list
    loading: false, // Indicates loading state
    loadingText: '', // Text displayed during loading
    trainingState: false, // Indicates if training is in progress
    currentTraining: "", // Currently training keyword
    trainedResults: {}, // Stores training results
    showTrainButton: false, // Controls visibility of the train button
    trainButtonText: '', // Text for the train button
    modelSaveSuccess: false, // Indicates if the model was saved successfully
    modelSaveError: false, // Indicates if there was an error saving the model
    returnedValAccuracy: 0, // Validation accuracy returned from the backend
    testFile: false, // Indicates if a test file is required
    test_tags: [], // Stores tags returned from the test file
    showTestTags: false, // Controls visibility of test tags
    runGAN: false, // Indicates if GAN training is required
    }
  },
  methods: {
    // Handles file upload and triggers keyword extraction
    onFileChange(file) {
    this.extractKeywords(file)
    },
    // Extracts keywords from the uploaded file
    extractKeywords(file){
    this.showUpload = false;
    this.loading = true;
    const data = new FormData();
    data.append('file', file);
    axios.post('/autotag/model/upload-pdf', data)
      .then(response => {
        this.keywords = response.data.keywords;
        this.selectedKeywords = Object.keys(response.data.keywords).filter(kw => response.data.keywords[kw].exists === false);
        this.loading = false;
        this.showKeywords = true;
      })
      .catch(error => {
        this.showUpload = true;
        this.loading = false;
      console.error('Error extracting keyword:', error);
      });
    },
    // Manually confirms and saves the model
    async manual_confirm_save(){
    const registerData = {
      template: "keras/MultiClassSingleTagKerasStandardModelTemplateA.py",
      group: this.currentTraining,
      model_key: this.currentTraining + '_model.zip',
    };
    axios.post('/autotag/model/register', registerData).then(response => {
      console.log("Model Registered!")
    });
    this.modelSaveSuccess = true;
    this.modelSaveError = false;
    },
    // Registers the model with the backend
    async registerModel(registerData) {
    this.loading = true;
    this.loadingText = 'Registering Model...';
    await axios.post('/autotag/model/register', registerData).then(response => {
      console.log("Model Registered!")
    });
    this.loading = false;
    },
    // Fetches training data and trains the model
    async fetchTrain(data) {
    try {
      this.loading = true;
      this.loadingText = 'Training...';
      const response = await axios.post('/autotag/img/fetch_train', data);
      console.log("Model Trained!", response.data); // Debugging
      return response.data; // Ensure the correct value is returned
    } catch (error) {
      this.modelSaveError = true;
      console.error('Error:', error);
      return undefined; // Explicitly return undefined in case of an error
    } finally {
      this.loading = false;
    }
    },
    // Prompts manual confirmation for saving the model
    promptManualConfirm() {
    return new Promise(resolve => {
      this.testFile = true;
      this.$watch('modelSaveSuccess', (newValue) => {
      if (newValue) {
        resolve(); // Ensure the promise resolves if modelSaveSuccess becomes true
      }
      });
    });
    },
    // Handles test file upload for manual confirmation
    onTestFileChange(file) {
    this.myfetchApi(file)
    },
    // Fetches tags for the uploaded test file
    myfetchApi(file) {
    const data = new FormData();
    data.append('img', file);
    data.append('temp_tag', this.currentTraining);
    axios.post('/autotag/tagTestVerify', data)
      .then(response => {
        this.showTestTags = true;
        this.test_tags = response.data.tags;
      })
      .catch(error => {
      console.error('Error fetching tags:', error);
      });
    },
    // Trains the selected keywords
    async trainKeywords() {
      this.showKeywords = false;
      this.trainingState = true;  
      for (const keyword of this.selectedKeywords) {
        this.testFile = false; 
        this.modelSaveSuccess = false;
        this.modelSaveError = false;
        this.showTestTags = false;
        this.showTrainButton = false;
        this.runGAN = false;
        this.currentTraining = keyword;
        const data = {
        tag: keyword,
        };
        const MAX_RETRIES = 2; // Set your desired maximum number of retries
        let retryCount = 0;
        while (true) {
        this.returnedValAccuracy = await this.fetchTrain(data);
        console.log('Returned Value:', this.returnedValAccuracy); // Debugging
        if (this.returnedValAccuracy > 0.86) {
          this.modelSaveSuccess = true;
          this.modelSaveError = false;
          const registerData = {
          template: "keras/MultiClassSingleTagKerasStandardModelTemplateA.py",
          group: this.currentTraining,
          model_key: this.currentTraining + '_model.zip',
          };
          await this.registerModel(registerData); 
          break;
        } 
        if (this.returnedValAccuracy >= 0.50 && this.returnedValAccuracy <= 0.85) {
          this.modelSaveError = false;
          await this.promptManualConfirm()
          break;
        } 
        this.modelSaveSuccess = false;
        this.modelSaveError = true;
        retryCount++;
        if (retryCount <= MAX_RETRIES) {
          continue;
        } else {
          this.modelSaveError = true;
          console.error('Maximum number of retries reached. Model not saved.');
          this.showTrainButton = true;
          this.trainButtonText = "Run GAN and Train";
          await this.promptGANandTrain();
        }
        break;
        }
        this.trainedResults[keyword] = this.returnedValAccuracy * 100; // Store the accuracy for each keyword
        this.currentTraining = null;
      }
    },
    // Prompts GAN training and retraining
    async promptGANandTrain() {
    return new Promise(async resolve => {
      this.runGAN = true;
      await this.runGANandTrain();
      if (this.modelSaveSuccess) {
      resolve();
      }
    });
    },
    // Trains the model using GAN-CNN
    async trainGanCnn(data) {
    try {
      const response = await axios.post('/autotag/ml/train/fetch_gan_cnn', data);
      return response.data;
    } catch (error) {
      console.error('Error training GAN-CNN:', error);
      throw error;
    }
    },
    // Runs GAN training and retraining logic
    async runGANandTrain() {
      const data = {
        tag: this.currentTraining,
      };
      this.loading = true;
      this.loadingText = 'Running GAN and Training...';
      const MAX_RETRIES = 2; // Set your desired maximum number of retries
      let retryCount = 0;
      while (true) {
        this.returnedValAccuracy = await this.trainGanCnn(data);
        console.log('Returned Value:', this.returnedValAccuracy); // Debugging
        if (this.returnedValAccuracy > 0.86) {
        this.modelSaveSuccess = true;
        this.modelSaveError = false;
        const registerData = {
          template: "keras/MultiClassSingleTagKerasStandardModelTemplateA.py",
          group: this.currentTraining,
          model_key: this.currentTraining + '_model.zip',
        };
        await this.registerModel(registerData); 
        break;
        } 
        if (this.returnedValAccuracy >= 0 && this.returnedValAccuracy <= 0.85) {
        this.modelSaveError = false;
        await this.promptManualConfirm()
        break;
        } 
        this.modelSaveSuccess = false;
        this.modelSaveError = true;
        retryCount++;
        if (retryCount <= MAX_RETRIES) {
        continue;
        } else {
        this.modelSaveError = true;
        console.error('Maximum number of retries reached. Model not saved.');
        this.showTrainButton = true;
        this.trainButtonText = "Run GAN and Train";
        }
        break;
      }
      this.loading = false;
    },
  }
  }
</script>
  
<style scoped>
  .animated-fade {
  animation: fadeInOut 0.75s infinite alternate;
  }

  @keyframes fadeInOut {
  0% { opacity: 0; }
  100% { opacity: 1; }
  }
</style>
