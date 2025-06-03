<template>
  <main style="background-color: #fcfaff; height:100vh;">
    <div>
    <!-- UploadFile component for uploading PDF files -->
    <UploadFile v-if="showUpload" fileType="application/pdf" title="PDF Uploader" :maxFileSizeMB="100" @newFile="extractKeywords($event)" />
    </div>
    <!-- Loader displayed while extracting keywords -->
    <ClipLoader :loading="loading && !showKeywordsSelect && !trainingState" size="100px" msg="Extracting Keywords..." style="margin: 100px auto;"/>
    
    <!-- Display extracted keywords -->
    <div v-if="showKeywordsSelect" style="font-size: larger; width: 50%; max-width: fit-content; margin: 100px auto;">
    <p style="font-size: 30px; font-weight: bold; color:#d574af; text-align: center;">Extracted Keywords: </p>
    <ul style="padding: 0; margin: 0; list-style-position: inside;">
      <li v-for="(info, keyword) in keywords" :key="keyword" style="list-style-type: none; font-size: 20px; text-align: center; margin-top: 10px;">
      <label :for="`kw-${keyword}`" style="cursor: pointer; color: #6c757d;">
        <b>{{ keyword }}</b> (score: {{ info.score }})
      </label>
      <!-- Show model existence status -->
      <span v-if="info.exists" style="margin-left: 10px; color: #28a745;" >: Model exists.</span>
      <!-- for checkbox selection of keywords
      <input v-else type="checkbox" :id="`kw-${keyword}`" :value="keyword" v-model="selectedKeywords" style="margin-left: 10px; transform: scale(1.2);"/>
      -->
      </li>
    </ul>
    <!-- Button to train selected keywords -->
    <button @click="trainKeywords" style="width: 150px; border: 2px solid #158be3; margin:30px; text-transform: none; border-radius: 5px; background-color: #158be3; box-shadow: 0 2px 4px white(0, 0, 0, 0.1);
      color: white; padding: 8px;">Start Training</button>
    </div>  
  
    <!-- Training state UI -->
    <div style=" display:flex; column-gap: 20px; margin: 100px auto;">
    <div v-if="showKeywordsState" style="flex: 0.5; white-space: nowrap; margin:20px auto">
      <p style="font-size: 30px; font-weight: bold; color:#d574af; text-align: center;">Keywords:</p>
      <ul style="padding: 0; margin: 0; list-style-position: inside;">
      <li v-for="(info, keyword) in keywords" :key="keyword" style="list-style-type: none; font-size: 20px; text-align: center; margin-top: 10px; white-space: nowrap;">
        <label :for="`kw-${keyword}`" style="cursor: pointer; color: #6c757d;">
        <b>{{ keyword }}</b>
        </label>
        <!-- Display training status -->
        <span v-if="info.exists" style="margin-left: 10px; color: #28a745;" >: Model already exists.</span>
        <span v-else-if="keyword === currentTraining" class="animated-fade">: Training...</span>
        <span v-else-if="trainedResults[keyword] == 'failed'" style="margin-left: 10px; color: #a72835;" >: Model training failed.</span>
        <span v-else-if="trainedResults[keyword] !== undefined" style="margin-left: 10px; color: #28a745;" >: Model trained, accuracy: {{ trainedResults[keyword] }}%</span>
        <span v-else>: Queued...</span>
      </li>
      </ul>
      <button v-if="currentTraining === null" @click="resetPage" style="width: 150px; border: 2px solid #158be3; margin:30px; text-transform: none; border-radius: 5px; background-color: #158be3; box-shadow: 0 2px 4px white(0, 0, 0, 0.1);
      color: white; padding: 8px;">Done</button>
    </div> 

    <!-- Training progress and manual confirmation -->
    <div v-if="trainingState" style="flex: 1; border:#158be3 2px solid; border-radius: 20px; padding: 20px; background-color: white; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
      <p style="font-size: 30px; font-weight: bold; color:#d574af; text-align: center;">Training "{{ this.currentTraining }}" Model...</p>
      <ClipLoader :loading="loading" size="100px" :msg="loadingText" style="margin: 100px auto;"/>

      <!-- Error message if model save fails -->
      <div v-if="modelSaveError">
      <p style="font-size: medium; font-weight: bold; color: rgb(227, 73, 73); ">
        Model was not saved due to lower validation accuracy ie: {{ returnedValAccuracy * 100 }}%. Re-Autotraining in Progress.
      </p>
      </div>
      <!-- user data input -->
      <div v-if="showUserDataUpload">
        <p style="font-size: medium; font-weight: bold; color: rgb(227, 73, 73); ">
          Couldn't train the model with sufficient accuracy, upload additional {{ this.currentTraining }} image data to train the model. 
        </p>
        <UploadFile fileType="application/zip,application/x-zip-compressed,application/octet-stream" title="ZIP Uploader" :maxFileSizeMB="500" @newFile="processUserData($event)" />
      </div>
      <!-- Manual testing UI -->
      <div v-if="testFile">
      <p style="font-size: medium; font-weight: bold; color: rgb(227, 73, 73); ">
        The model's accuracy was in between 75 to 90. 
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
      <button @click="manualChoice = 'save'" style="width: auto;
        border: 2px solid #158be3;
        text-transform: none;
        margin-right:20px;
        border-radius: 20px  20px  20px 20px;
        background-color: #158be3;
        box-shadow: 0 2px 4px white(0, 0, 0, 0.1);
        color: white; padding: 8px;">Save Model</button>
        <button @click="manualChoice = 'retrain'" style="width: auto;
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
    showKeywordsSelect: false, // Controls visibility of the keywords list
    showKeywordsState: false, // Controls visibility of the keywords state
    loading: false, // Indicates loading state
    loadingText: '', // Text displayed during loading
    trainingState: false, // Indicates if training is in progress
    currentTraining: "", // Currently training keyword
    trainedResults: {}, // Stores training results
    modelSaveSuccess: false, // Indicates if the model was saved successfully
    modelSaveError: false, // Indicates if there was an error saving the model
    returnedValAccuracy: 0, // Validation accuracy returned from the backend
    testFile: false, // Indicates if a test file is required
    test_tags: [], // Stores tags returned from the test file
    showTestTags: false, // Controls visibility of test tags
    manualChoice: null,    // will hold "save" or "retrain"
    showUserDataUpload: false, // Indicates if user data input is required
    }
  },
  methods: {
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
        this.showKeywordsSelect = true;
      })
      .catch(error => {
        this.showUpload = true;
        this.loading = false;
      console.error('Error extracting keyword:', error);
      });
    },
    // Registers the model with the backend
    async registerModel() {
    this.loading = true;
    this.loadingText = 'Registering Model...';
    const registerData = {
      template: "keras/MultiClassSingleTagKerasStandardModelTemplateA.py",
      group: this.currentTraining,
      model_key: this.currentTraining + '_model.zip',
    };
    await axios.post('/autotag/model/register', registerData)
    .then(response => {
      console.log("Model Registered!")
    })
    .catch(error => {
      console.error('Error processing registering model:', error);
    })
    .finally(() => {
      this.loading = false;
    });
    },
    // Fetches training data and trains the model
    async fetchTrain(data) {
    try {
      this.loading = true;
      this.loadingText = 'Training...';
      const response = await axios.post('/autotag/img/fetch_train', data);
      return response.data;
    } catch (error) {
      this.modelSaveError = true;
      console.error('Error:', error);
      return undefined; // Explicitly return undefined in case of an error
    } finally {
      this.loading = false;
    }
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
    // Handles manual decision for saving or retraining the model
    async manualDecision() {
      return new Promise(resolve => {
        this.$watch(
          () => this.manualChoice,
          (newVal) => {
            if (newVal) {
              resolve(newVal);
            }
          }
        );
      });
    },
    // Resets flags for training state
    resetFlags() {
      this.testFile = false;
      this.modelSaveSuccess = false;
      this.modelSaveError = false;
      this.showTestTags = false;
      this.manualChoice = null;
    },
    // Trains the selected keywords
    async trainKeywords() {
      this.showKeywordsSelect = false;
      this.trainingState = true; 
      this.showKeywordsState = true;

      for (const keyword of this.selectedKeywords) {
        this.resetFlags(); // Reset flags for each keyword
        this.currentTraining = keyword;
        console.log('Training keyword:', keyword); // Debugging
        const data = {
        tag: keyword,
        };
        const MAX_RETRIES = 1; // Set your desired maximum number of retries
        let retryCount = 0;
        while (true) {
        this.returnedValAccuracy = await this.fetchTrain(data);
        console.log('Returned Value:', this.returnedValAccuracy); // Debugging
        if (this.returnedValAccuracy > 0.90) {
          this.modelSaveSuccess = true;
          this. modelSaveError= false;
          await this.registerModel(); 
          break;
        } 
        else if (this.returnedValAccuracy >= 0.75) {
          this.modelSaveError = false;
          this.testFile = true;
          const choice = await this.manualDecision();
          if (choice === "save") {
            // save path
            await this.registerModel();
            this.modelSaveSuccess = true;
            break;
          } else {
            this.testFile = false;
            this.showTestTags = false;
            this.manualChoice = null;
          }
        }
        else {
          this.modelSaveSuccess = false;
          this.modelSaveError = true;
        }
        retryCount++;
        console.log('continued in fetchtrain, retryCount:', retryCount); // Debugging
        if (retryCount <= MAX_RETRIES) {
          continue;
        } else {
          this.modelSaveError = false;
          console.error('Maximum number of retries reached. Model not saved.');
          await this.runGANandTrain();
        }
        break;
        }
        if (!this.trainedResults[keyword]) {
          this.trainedResults[keyword] = (parseFloat(this.returnedValAccuracy) * 100).toFixed(2); // Store the accuracy for each keyword with 2 decimal places
        }
      }
      this.trainingState = false; // Reset training state
      this.currentTraining = null;
      this.showKeywordsState = true;
      
    },

    // Trains the model using GAN-CNN
    async trainGanCnn(data) {
      this.loading = true;
      this.loadingText = 'Running GAN and Training...';
    try {
      const response = await axios.post('/autotag/ml/train/fetch_gan_cnn', data);
      return response.data;
    } catch (error) {
      console.error('Error training GAN-CNN:', error);
      throw error;
    } finally {
      this.loading = false;
    }
    },
    // Runs GAN training and retraining logic
    async runGANandTrain() {
      const data = {
        tag: this.currentTraining,
      };
      const MAX_RETRIES = 1; // Set your desired maximum number of retries
      let retryCount = 0;
      while (true) {
        this.returnedValAccuracy = await this.trainGanCnn(data);
        if (this.returnedValAccuracy > 0.90) {
        this.modelSaveSuccess = true;
        this.modelSaveError = false;
        await this.registerModel(); 
        break;
        } 
        else if (this.returnedValAccuracy >= 0.75) {
          this.modelSaveError = false;
          this.testFile = true;
          const choice = await this.manualDecision();
          if (choice === "save") {
            // save path
            await this.registerModel();
            this.modelSaveSuccess = true;
            break;
          } else {
            this.testFile = false;
            this.showTestTags = false;
            this.manualChoice = null;
          }
        }
        else {
          this.modelSaveSuccess = false;
          this.modelSaveError = true;
        }
        retryCount++;
        console.log('continued in gan, retryCount:', retryCount); // Debugging
        if (retryCount <= MAX_RETRIES) {
        continue;
        } else {
        this.modelSaveError = false;
        await this.userDataInput(); // Call user data input function
        }
        break;
      }
      this.loading = false;
    },
     processUserData(file) {
      console.log('Processing user data:'); // Debugging
      this.showUserDataUpload = false;
      this.loading = true;
      this.loadingText = 'Training with User Data...';
      this.returnedValAccuracy = null;
      const data = new FormData();
      data.append('tag', this.currentTraining);
      data.append('dataset', file);
      data.append('use_crawled_data', true);
      axios.post('/autotag/img/fetch_train_userdata', data)
        .then(response => {
          this.returnedValAccuracy = response.data.accuracy; 
        })
        .catch(error => {
          console.error('Error processing user data:', error);
        })
        .finally(() => {
          this.loading = false;
        });
    },
    async userDataInput() {
      // initialize
      this.loading = false;
      this.showUserDataUpload = true;
      this.modelSaveSuccess = false;
      this.modelSaveError = false;

      return new Promise(resolve => {
        // watch accuracy **once per upload**
        const unwatch = this.$watch(
          () => this.returnedValAccuracy,
          async () => {
            // ignore null 
            if (this.returnedValAccuracy == null) {
            return;
          }
            // hide the uploader
            this.showUserDataUpload = false;
            if (this.returnedValAccuracy > 0.90) {
              this.modelSaveError = false;
              this.modelSaveSuccess = true;
              await this.registerModel();
              unwatch();
              resolve();
            }
            else if (this.returnedValAccuracy >= 0.75) {
              this.modelSaveError = false;
              this.testFile = true;
              const choice = await this.manualDecision();

              if (choice === 'save') {
                // user chose save
                await this.registerModel();
                this.modelSaveSuccess = true;
                unwatch();
                resolve();
              }
              else {
                // user chose retrain:
                this.loading = false;
                this.testFile = false;
                this.showTestTags = false;
                this.manualChoice = null;
                this.showUserDataUpload = true;   // 🔁 show uploader again
                // **don’t** resolve yet – leave the watcher active
              }
            }
            else {
              // accuracy too low
              this.modelSaveSuccess = false;
              this.modelSaveError = true;
              this.trainedResults[this.currentTraining] = 'failed'; 
              unwatch();
              resolve();
            }

            if (!this.trainedResults[this.currentTraining]) {
              this.trainedResults[this.currentTraining] = (parseFloat(this.returnedValAccuracy) * 100).toFixed(2); // Store the accuracy for each keyword with 2 decimal places
            }

            this.loading = false;
          },
          { immediate: false }
        );
      });
    },

    // Resets the page state
    resetPage() {
    this.keywords= {}; // Stores extracted keywords
    this.selectedKeywords= []; // Stores selected keywords for training
    this.showUpload= true; // Controls visibility of the upload component
    this.showKeywordsSelect= false; // Controls visibility of the keywords list
    this.showKeywordsState= false; // Controls visibility of the keywords state
    this.loading= false; // Indicates loading state
    this.loadingText= ''; // Text displayed during loading
    this.trainingState= false; // Indicates if training is in progress
    this.currentTraining= ""; // Currently training keyword
    this.trainedResults= {}; // Stores training results
    this.modelSaveSuccess= false; // Indicates if the model was saved successfully
    this.modelSaveError= false; // Indicates if there was an error saving the model
    this.returnedValAccuracy= 0; // Validation accuracy returned from the backend
    this.testFile= false; // Indicates if a test file is required
    this.test_tags= []; // Stores tags returned from the test file
    this.showTestTags= false; // Controls visibility of test tags
    this.manualChoice= null;    // will hold "save" or "retrain"
    this.showUserDataUpload= false; // Indicates if user data input is required
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
