<template>
  <main class="page-container">
    <div class="content-wrapper">
      <!-- Initial Upload State -->
      <div v-if="!trainingState && !modelSaveSuccess && !finalFailure" class="panel animated-fade-in">
        <UploadFile
          fileType="image/*"
          title="Image Uploader"
          :maxFileSizeMB="5"
          @newFile="onFileChange($event)"
        />
        <ClipLoader :loading="loading" size="60px" msg="Analyzing image..." style="margin-top: 20px;" />
      </div>

      <!-- New Model Configuration State -->
      <div v-if="showStringField" class="panel animated-fade-in">
        <h2 class="panel__title">New Model Configuration</h2>
        <p class="panel__description">This image was not recognized. Provide a label and set the success thresholds to train a new model.</p>
        
        <div class="input-group">
          <input v-model="customString" placeholder="Enter image label (e.g., 'Golden Retriever')" class="form-input" @keyup.enter="startTraining" />
        </div>
        
        <div class="threshold-section">
          <h3 class="section-header">Success Thresholds</h3>
          <div class="sliders-container">
            <div v-for="(value, metric) in thresholds" :key="metric">
              <label :for="`slider-${metric}`" class="slider-label">
                {{ metric }} Threshold: <b>{{ value }}%</b>
              </label>
              <input type="range" :id="`slider-${metric}`" :min="manualCheckBaseline" max="99" v-model.number="thresholds[metric]" class="slider" />
            </div>
          </div>
          <p class="info-box">
            Models that meet all thresholds will be saved automatically. Models with any metric below a threshold but above <b>{{ manualCheckBaseline }}%</b> will require manual review.
          </p>
        </div>
        
        <button @click="startTraining" :disabled="!customString" class="btn btn--primary btn--full-width">Start Training</button>
      </div>

      <!-- Model Found State -->
      <div v-if="showTags" class="panel animated-fade-in">
         <h2 class="panel__title">Model Found!</h2>
         <p class="panel__description">This image was identified with the following label:</p>
         <div class="tags-display">{{ tags }}</div>
         <button @click="resetPage" class="btn btn--primary" style="margin-top: 25px;">Upload Another</button>
      </div>

      <!-- Training In Progress State -->
      <div v-if="trainingState" class="panel animated-fade-in">
        <h2 class="panel__title">Training Model for "{{ this.customString }}"</h2>
        <ClipLoader :loading="loading && !testFile && !showUserDataUpload" size="80px" :msg="loadingText" style="margin: 40px auto;" />

        <div v-if="modelSaveError" class="alert alert--error">
          <p>The '{{ loadingText.split(':')[1] || 'current' }}' strategy did not succeed. Attempting next strategy...</p>
        </div>

        <div v-if="showUserDataUpload" class="action-box">
          <h3 class="action-box__title">Final Attempt: Manual Data Upload</h3>
          <p class="action-box__description">
            The automated strategies failed. Please upload a ZIP file containing correctly labeled images to train the model with your own data.
          </p>
          <UploadFile
            fileType="application/zip,application/x-zip-compressed"
            title="Upload ZIP with Training Images"
            :maxFileSizeMB="500"
            @newFile="processUserData($event)"
          />
        </div>

        <div v-if="testFile" class="action-box">
          <h3 class="action-box__title">Manual Confirmation Needed</h3>
          <div v-if="failingMetrics.length > 0" class="alert alert--warning">
            <p class="alert__title">Some metrics were below the automatic success threshold:</p>
            <ul class="metric-list">
              <li v-for="metric in failingMetrics" :key="metric.name" class="metric-list__item">
                <span class="metric-name">{{ metric.name }}:</span>
                <span class="metric-value metric-value--actual">{{ (metric.value * 100).toFixed(1) }}%</span>
                <span class="metric-value metric-value--threshold">(Threshold: {{ metric.threshold }}%)</span>
              </li>
            </ul>
          </div>
          <p class="action-box__description">Please upload a test image to verify the model's prediction.</p>
          <UploadFile v-if="!loading" fileType="image/*" title="Upload Manual Test Image" :maxFileSizeMB="5" @newFile="onTestFileChange($event)" />
          <ClipLoader :loading="loading" size="60px" :msg="loadingText" style="margin: 20px auto;" />
        </div>
        
        <div v-if="showTestTags" class="action-box">
          <h3 class="action-box__title">Test Result</h3>
          <div class="test-result">
            <span>Predicted Label:</span>
            <span class="test-result__tags">{{ test_tags || 'None' }}</span>
          </div>
          <p class="action-box__description" style="margin-top: 20px;">Do you want to save this model or proceed to the next training strategy?</p>
          <div class="button-group">
            <button @click="manualChoice = 'save'" class="btn btn--primary">Save Model</button>
            <button @click="manualChoice = 'retrain'" class="btn btn--secondary">Re-Train</button>
          </div>
        </div>
      </div>

      <!-- Final Success/Failure States -->
      <div v-if="modelSaveSuccess" class="panel animated-fade-in">
        <div class="final-status final-status--success">✓</div>
        <h2 class="panel__title">Training Complete!</h2>
        <p class="panel__description">The model for "<b>{{ customString }}</b>" was successfully trained and saved.</p>
        <button @click="resetPage" class="btn btn--primary">Train Another Model</button>
      </div>

      <div v-if="finalFailure" class="panel animated-fade-in">
        <div class="final-status final-status--failed">✗</div>
        <h2 class="panel__title">Training Failed</h2>
        <p class="panel__description">All training strategies for "<b>{{ customString }}</b>" failed to meet the required thresholds.</p>
        <button @click="resetPage" class="btn btn--secondary">Start Over</button>
      </div>
    </div>
  </main>
</template>

<script>
// The <script> block from the original component remains unchanged.
// It contains the robust logic and does not need modification.
import UploadFile from '@/components/UploadFile.vue';
import ClipLoader from '@/components/ClipLoader.vue';
import axios from 'axios';

export default {
  name: 'Page3ViewRefactored',
  components: { UploadFile, ClipLoader },
  data() {
    return {
      // Core state
      tags: [],
      customString: '',

      // UI Visibility
      loading: false,
      loadingText: '',
      trainingState: false, 
      showStringField: false,
      showTags: false,

      // Configurable thresholds
      thresholds: {
        accuracy: 95,
        f1: 88,
        recall: 85,
        precision: 85,
      },
      manualCheckBaseline: 60,

      // State for a training cycle
      returnedMetrics: {},
      failingMetrics: [],
      modelSaveSuccess: false,
      finalFailure: false,
      modelSaveError: false,
      testFile: false,
      showTestTags: false,
      test_tags: [],
      manualChoice: null,
      
      // State for User Data Upload step
      showUserDataUpload: false,
      userDataPromiseResolver: null,
    };
  },
  methods: {
    onFileChange(file) {
      this.loading = true;
      const data = new FormData();
      data.append('img', file);

      axios.post('/tag', data)
        .then(response => {
          if (response.data.tags === "We dont have model to recognise this, please train it first") {
            this.showStringField = true;
          } else {
            this.showTags = true;
            this.tags = response.data.tags;
          }
        })
        .catch(error => {
          console.error('Error fetching tags:', error);
          this.showStringField = true;
        })
        .finally(() => { this.loading = false; });
    },

    async startTraining() {
      if (!this.customString) return;
      this.trainingState = true;
      this.showStringField = false;
      
      const resultStatus = await this.runTrainingPipeline();
      
      this.trainingState = false;
      if (resultStatus === 'success') {
        this.modelSaveSuccess = true;
      } else {
        this.finalFailure = true;
      }
    },
    
    async runTrainingPipeline() {
      const strategies = [
        { name: 'Standard Training', method: this.fetchTrain, args: { tag: this.customString } },
        { name: 'GAN + CNN Training', method: this.trainGanCnn, args: { tag: this.customString } },
        { name: 'Manual Data Upload', method: this.userDataInput, args: {} }
      ];

      for (const strategy of strategies) {
        this.resetFlagsForNewStrategy();
        this.loadingText = strategy.name === 'Manual Data Upload' ? 'Awaiting User Data...' : `Running: ${strategy.name}...`;

        const metrics = await strategy.method(strategy.args);
        if (!metrics || typeof metrics !== 'object') {
          this.modelSaveError = true;
          await new Promise(r => setTimeout(r, 2500));
          continue;
        };
      
        this.returnedMetrics = metrics;
        const outcome = this.evaluateMetrics(metrics);

        if (outcome === 'success') {
          await this.registerModel();
          return 'success';
        }

        if (outcome === 'manual') {
          const choice = await this.performManualConfirmation();
          this.resetFlagsForNewStrategy();

          if (choice === 'save') {
            await this.registerModel();
            return 'success';
          }
        }
        
        this.modelSaveError = true;
        await new Promise(r => setTimeout(r, 2500));
      }
      return 'failed';
    },

    evaluateMetrics(metrics) {
        this.failingMetrics = [];
        let meetsSuccessThreshold = true;
        let meetsManualBaseline = true;

        for (const metricName in this.thresholds) {
            const metricValue = metrics[metricName] || 0;
            const successThreshold = this.thresholds[metricName];

            if (metricValue * 100 < successThreshold) {
                meetsSuccessThreshold = false;
                this.failingMetrics.push({ name: metricName, value: metricValue, threshold: successThreshold });
            }

            if (metricValue * 100 < this.manualCheckBaseline) {
                meetsManualBaseline = false;
            }
        }

        if (meetsSuccessThreshold) return 'success';
        if (meetsManualBaseline) return 'manual';
        return 'failure';
    },

    async performManualConfirmation() {
      return new Promise(resolve => {
        this.testFile = true;
        
        const unwatch = this.$watch(() => this.manualChoice, (choice) => {
          if (choice) {
            unwatch();
            resolve(choice);
          }
        });
      });
    },
    
    async onTestFileChange(file) {
      this.loading = true;
      this.loadingText = 'Verifying Test Image...';
      const data = new FormData();
      data.append('img', file);
      data.append('temp_tag', this.customString);

      try {
        const response = await axios.post('/autotag/tagTestVerify', data);
        this.test_tags = response.data.tags;
        this.showTestTags = true;
      } catch (error) {
        console.error('Error fetching test tags:', error);
        this.test_tags = "Error during prediction.";
        this.showTestTags = true;
      } finally {
        this.loading = false;
      }
    },
    
    // --- Methods for User Data Upload Strategy ---
    async userDataInput() {
      this.loading = false;
      this.showUserDataUpload = true;
      return new Promise(resolve => {
        this.userDataPromiseResolver = resolve;
      });
    },

    processUserData(file) {
      this.showUserDataUpload = false;
      this.loading = true;
      this.loadingText = 'Training with User Data...';
      const data = new FormData();
      data.append('tag', this.customString);
      data.append('dataset', file);
      
      axios.post('/autotag/img/fetch_train_userdata', data)
        .then(response => {
          if (this.userDataPromiseResolver) {
  
            this.userDataPromiseResolver(response.data);
          }
        })
        .catch(error => {
          console.error('Error processing user data:', error);
          if (this.userDataPromiseResolver) {
            this.userDataPromiseResolver(undefined);
          }
        })
        .finally(() => {
          this.loading = false;
        });
    },

    // --- API & HELPER METHODS ---
    async fetchTrain(data) {
      this.loading = true;
      try {
        const response = await axios.post('/autotag/img/fetch_train', data);
        return response.data;
      } catch (error) { console.error('Error in fetchTrain:', error); return undefined; }
      finally { this.loading = false; }
    },

    async trainGanCnn(data) {
      this.loading = true;
      try {
        const response = await axios.post('/autotag/ml/train/fetch_gan_cnn', data);
        return response.data;
      } catch (error) { console.error('Error in GAN-CNN:', error); return undefined; }
      finally { this.loading = false; }
    },

    async registerModel() {
        this.loading = true;
        this.loadingText = 'Saving Model...';
        try {
            await axios.post('/autotag/model/register', {
                template: "keras/MultiClassSingleTagKerasStandardModelTemplateA.py",
                group: this.customString,
                model_key: this.customString + '_model.zip',
            });
            console.log("Model Registered!");
        } catch (error) { console.error('Error registering model:', error); }
        finally { this.loading = false; }
    },
    
    resetFlagsForNewStrategy() {
      this.modelSaveError = false;
      this.testFile = false;
      this.showTestTags = false;
      this.manualChoice = null;
      this.failingMetrics = [];
      this.showUserDataUpload = false;
    },

    resetPage() {
      Object.assign(this.$data, this.$options.data.apply(this));
    }
  }
}
</script>

<style scoped>

/* --- Main Layout & Font --- */
.page-container {
  background-color: #f8f9fa; /* Unified light background */
  min-height: 100vh;
  padding: 40px 20px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  color: #343a40;
}
.content-wrapper {
  max-width: 800px; /* Standardized max-width */
  margin: 0 auto;
}
.animated-fade-in {
  animation: fadeIn 0.6s ease-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(-15px); }
  to { opacity: 1; transform: translateY(0); }
}

/* --- Panels & Headers --- */
.panel {
  width: 100%;
  margin: 20px auto;
  padding: 30px 40px;
  background-color: #ffffff;
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05); /* Softer shadow */
  border: 1px solid #e9ecef;
  text-align: center;
}
.panel__title {
  font-size: 26px;
  font-weight: 700;
  color: #212529;
  margin: 0 0 10px 0;
}
.panel__description {
  font-size: 16px;
  color: #6c757d;
  margin: 0 auto 30px auto;
  max-width: 550px;
  line-height: 1.6;
}
.section-header {
  font-size: 18px;
  font-weight: 600;
  color: #495057;
  text-align: left;
  margin: 0 0 15px 0;
  padding-bottom: 10px;
  border-bottom: 1px solid #e9ecef;
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

/* --- Forms & Inputs --- */
.input-group {
  margin-bottom: 30px;
}
.form-input {
  width: 100%;
  border: 1px solid #ced4da;
  padding: 14px 18px;
  font-size: 16px;
  border-radius: 10px;
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.form-input:focus {
  border-color: #158be3;
  box-shadow: 0 0 0 3px rgba(21, 139, 227, 0.15);
}

/* --- Sliders --- */
.threshold-section {
  text-align: left;
  margin-bottom: 30px;
  border-top: 1px solid #f1f3f5;
  padding-top: 30px;
}
.sliders-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 25px;
}
.slider-label {
  display: block;
  font-size: 15px;
  font-weight: 600;
  margin-bottom: 12px;
  text-transform: capitalize;
  color: #495057;
}
.slider-label b {
  color: #158be3;
  float: right;
  font-size: 16px;
}
.slider {
  width: 100%;
  cursor: pointer;
  -webkit-appearance: none;
  appearance: none;
  height: 10px;
  background: #e9ecef;
  border-radius: 5px;
  outline: none;
}
.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 22px;
  height: 22px;
  background: #158be3;
  border-radius: 50%;
  border: 3px solid white;
  box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}
.slider::-moz-range-thumb {
  width: 22px;
  height: 22px;
  background: #158be3;
  border-radius: 50%;
  border: 3px solid white;
  box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}

/* --- Alerts & Info Boxes --- */
.info-box {
  font-size: 14px;
  color: #6c757d;
  margin-top: 30px;
  text-align: center;
  background-color: #f8f9fa;
  padding: 15px;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}
.alert {
  padding: 15px 20px;
  border-radius: 10px;
  margin: 25px 0;
  text-align: left;
  border: 1px solid transparent;
}
.alert__title {
  font-weight: 700;
  margin: 0 0 8px 0;
}
.alert--error {
  background-color: #fbebed;
  border-color: #f5c6cb;
  color: #842029;
}
.alert--error .alert__title { color: #dc3545; }
.alert--warning {
  background-color: #fff3cd;
  border-color: #ffeeba;
  color: #856404;
}
.alert--warning .alert__title { color: #856404; }

/* --- Component-Specific Styles --- */
.tags-display {
  font-size: 20px;
  font-weight: bold;
  padding: 10px 20px;
  border-radius: 8px;
}

.action-box {
  padding: 25px;
  border: 1px solid #e9ecef;
  border-radius: 12px;
  background-color: #f8f9fa;
  margin-top: 25px;
  text-align: center;
}
.action-box__title {
  font-size: 18px;
  font-weight: 600;
  color: #343a40;
  margin: 0 0 10px 0;
}
.action-box__description {
  font-size: 15px;
  color: #6c757d;
  margin: 0 auto 20px auto;
  max-width: 450px;
}
.metric-list {
  list-style-type: none;
  padding-left: 0;
  margin: 10px 0 0 0;
  font-size: 15px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.metric-list__item {
  display: flex;
  align-items: center;
  gap: 10px;
  background-color: #fff9e6;
  padding: 8px 12px;
  border-radius: 6px;
}
.metric-name { text-transform: capitalize; font-weight: 600; min-width: 80px; }
.metric-value { font-weight: 600; }
.metric-value--actual { color: #d9480f; }
.metric-value--threshold { font-size: 14px; color: #6c757d; font-weight: 500;}

.test-result {
  background: white;
  padding: 15px;
  border-radius: 8px;
  font-size: 16px;
  display: flex;
  justify-content: center;
  align-items: baseline;
  gap: 10px;
}
.test-result__tags {
  font-weight: bold;
  font-size: 18px;
  color: #158be3;
}

.final-status {
  font-size: 50px;
  width: 90px;
  height: 90px;
  line-height: 90px;
  border-radius: 50%;
  margin: 0 auto 20px auto;
  font-weight: bold;
}
.final-status--success {
  background-color: #eaf6ec;
  color: #28a745;
}
.final-status--failed {
  line-height: 85px; /* Adjust for better centering of X */
  background-color: #fbebed;
  color: #dc3545;
}
</style>