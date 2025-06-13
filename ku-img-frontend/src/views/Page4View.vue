<template>
  <main class="page-container">
    <div class="content-wrapper" style="max-width: 1200px;">

      <div v-if="showUpload" class="panel animated-fade-in" style="max-width: 800px;">
        <UploadFile fileType="application/pdf" title="PDF Uploader" :maxFileSizeMB="100"
          @newFile="extractKeywords($event)" />
      </div>

      <ClipLoader :loading="loading && !showKeywordsSelect && !trainingState" size="100px"
        msg="Extracting Keywords from PDF..." style="margin: 100px auto;" />

      <div v-if="showKeywordsSelect" class="panel animated-fade-in" style="max-width: 800px;">
        <h2 class="panel__title">Training Configuration</h2>
        <p class="panel__description">The following keywords were extracted from the PDF. Models will be trained for all
          new keywords. You can adjust the success thresholds below.</p>

        <div class="keyword-list-section">
          <h3 class="section-header">Extracted Keywords</h3>
          <ul class="keyword-list">
            <li v-for="(info, keyword) in keywords" :key="keyword" class="keyword-item"
              :class="{ 'exists': info.exists }">
              <span class="keyword-name">{{ keyword }}</span>
              <span class="keyword-score">(score: {{ info.score }})</span>
              <span v-if="info.exists" class="status-badge status--exists">Model exists</span>
            </li>
          </ul>
        </div>

        <div class="threshold-section">
          <h3 class="section-header">Set Success Thresholds</h3>
          <div class="sliders-container">
            <div v-for="(value, metric) in thresholds" :key="metric">
              <label :for="`slider-${metric}`" class="slider-label">
                {{ metric }} Threshold: <b>{{ value }}%</b>
              </label>
              <input type="range" :id="`slider-${metric}`" :min="manualCheckBaseline" max="99"
                v-model.number="thresholds[metric]" class="slider" />
            </div>
          </div>
          <p class="info-box">
            Models that meet all thresholds will be saved automatically. Models with any metric below a threshold but
            above <b>{{ manualCheckBaseline }}%</b> will require manual review.
          </p>
        </div>

        <button @click="trainKeywords" class="btn btn--primary" style="padding: 14px 40px; margin-top: 20px;">Start
          Training for {{ selectedKeywords.length }} Keywords</button>
      </div>

      <div v-if="showKeywordsState || trainingState" class="training-dashboard animated-fade-in">
        <div class="panel status-panel">
          <h2 class="panel__title">Training Status</h2>
          <div class="status-grid">
            <template v-for="(info, keyword) in keywords" :key="keyword">

              <div class="status-grid__row">
                <div class="status-grid__cell status-grid__cell--keyword"><b>{{ keyword }}</b></div>
                <div class="status-grid__cell status-grid__cell--status">
                  <span v-if="trainedResults[keyword]?.status !== 'success'" class="status-badge"
                    :class="getKeywordClass(keyword, info)">
                    {{ getKeywordStatus(keyword, info) }}
                  </span>
                  <div v-else class="success-details">
                    <span class="status-badge" :class="getKeywordClass(keyword, info)">
                      {{ getKeywordStatus(keyword, info) }}
                    </span>
                    <div class="metrics-container">
                      <div v-for="(value, metric) in trainedResults[keyword].metrics" :key="metric" class="metric-row">
                        <span class="metric-label">{{ metric }}</span>
                        <div class="progress-bar-container">
                          <div class="progress-bar" :style="{ width: value * 100 + '%' }"></div>
                        </div>
                        <span class="metric-value">{{ (value * 100).toFixed(1) }}%</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div v-if="trainingState && currentTraining === keyword" class="training-panel-nested animated-fade-in">
                <h2 class="panel__title" style="font-size: 22px; margin-bottom: 20px;">Training Model for "{{
                  this.currentTraining }}"</h2>
                <ClipLoader :loading="loading && !testFile" size="80px" :msg="loadingText" style="margin: 40px auto;" />

                <div v-if="modelSaveError" class="alert alert--error">
                  <p>Strategy failed. Attempting next.</p>
                </div>

                <div v-if="showUserDataUpload" class="action-box">
                  <h3 class="action-box__title">Additional Data Required</h3>
                  <p class="action-box__description">
                    Upload a ZIP file with additional images for "{{ this.currentTraining }}" to improve the model.
                  </p>
                  <UploadFile fileType="application/zip,application/x-zip-compressed,application/octet-stream"
                    title="ZIP Uploader" :maxFileSizeMB="500" @newFile="processUserData($event)" />
                </div>

                <div v-if="testFile" class="action-box">
                  <h3 class="action-box__title">Manual Confirmation Needed</h3>
                  <div class="alert alert--warning">
                    <p class="alert__title">Metrics Below Threshold:</p>
                    <ul class="metric-list">
                      <li v-for="metric in failingMetrics" :key="metric.name" class="metric-list__item">
                        <span class="metric-name">{{ metric.name }}:</span>
                        <span class="metric-value metric-value--actual">{{ (metric.value * 100).toFixed(1) }}%</span>
                        <span class="metric-value metric-value--threshold">(Threshold: {{ metric.threshold }}%)</span>
                      </li>
                    </ul>
                  </div>
                  <p class="action-box__description">Please upload a test image to verify its predictions.</p>
                  <UploadFile v-if="!loading" fileType="image/*" title="Manual Test Image" :maxFileSizeMB="5"
                    @newFile="onTestFileChange($event)" />
                  <ClipLoader :loading="loading" size="60px" :msg="loadingText" style="margin: 20px auto;" />
                </div>

                <div v-if="showTestTags" class="action-box">
                  <h3 class="action-box__title">Test Result</h3>
                  <div class="test-result">
                    <span>Predicted Tags:</span>
                    <span class="test-result__tags">{{ test_tags.join(', ') || 'None' }}</span>
                  </div>
                  <p class="action-box__description" style="margin-top: 20px;">Based on this result, would you like to
                    save this model or re-train?</p>
                  <div class="button-group">
                    <button @click="manualChoice = 'save'" class="btn btn--primary">Save Model</button>
                    <button @click="manualChoice = 'retrain'" class="btn btn--secondary">Re-Train</button>
                  </div>
                </div>
              </div>
            </template>
          </div>
          <button v-if="currentTraining === null && showKeywordsState" @click="resetPage" class="btn btn--primary"
            style="margin-top: 30px;">Done</button>
        </div>
      </div>
    </div>
  </main>
</template>

<script>
// The <script> block remains unchanged from the original.
import UploadFile from '@/components/UploadFile.vue'
import TestUploadFile from '@/components/UploadTestFile.vue'
import ClipLoader from '@/components/ClipLoader.vue'
import axios from 'axios'

export default {
  name: 'Page4View',
  components: { UploadFile, TestUploadFile, ClipLoader },
  data() {
    return {
      // Core state
      keywords: {},
      selectedKeywords: [],
      currentTraining: "",
      trainedResults: {},

      // UI Visibility Flags
      showUpload: true,
      showKeywordsSelect: false,
      showKeywordsState: false,
      loading: false,
      loadingText: '',
      trainingState: false,

      // NEW: Thresholds object for multiple sliders
      thresholds: {
        accuracy: 95,
        f1: 88,
      },
      manualCheckBaseline: 60, // Fixed baseline for manual checks

      // State for a single keyword's training cycle
      returnedMetrics: {},
      failingMetrics: [], // To list metrics that failed
      modelSaveSuccess: false,
      modelSaveError: false,
      testFile: false,
      showTestTags: false,
      test_tags: [],
      manualChoice: null,
      showUserDataUpload: false,
      userDataPromiseResolver: null,
    }
  },
  methods: {
    // EXTRACT KEYWORDS
    extractKeywords(file) {
      this.showUpload = false;
      this.loading = true;
      const data = new FormData();
      data.append('file', file);
      axios.post('/autotag/model/upload-pdf', data)
        .then(response => {
          this.keywords = response.data.keywords;
          this.selectedKeywords = Object.keys(response.data.keywords).filter(kw => !response.data.keywords[kw].exists);
          this.loading = false;
          this.showKeywordsSelect = true;
        })
        .catch(error => {
          this.showUpload = true;
          this.loading = false;
          console.error('Error extracting keyword:', error);
        });
    },

    async trainKeywords() {
      this.showKeywordsSelect = false;
      this.trainingState = true;
      this.showKeywordsState = true;

      for (const keyword of this.selectedKeywords) {
        this.resetFlagsForNewKeyword();
        this.currentTraining = keyword;

        const resultStatus = await this.runTrainingPipeline();

        // Store comprehensive results
        this.trainedResults[keyword] = {
          status: resultStatus,
          metrics: { ...this.returnedMetrics }
        };
      }
      this.currentTraining = null;
      this.trainingState = false;
    },

    async runTrainingPipeline() {
      const strategies = [
        { name: 'Standard Training', method: this.fetchTrain, args: { tag: this.currentTraining } },
        { name: 'GAN + CNN Training', method: this.trainGanCnn, args: { tag: this.currentTraining } },
        { name: 'Manual Data Upload', method: this.userDataInput, args: {} }
      ];

      for (const strategy of strategies) {
        this.resetFlagsForNewStrategy();
        this.loadingText = `Running: ${strategy.name}...`;

        const metrics = await strategy.method(strategy.args);
        if (!metrics || typeof metrics !== 'object') continue;

        this.returnedMetrics = metrics;
        const outcome = this.evaluateMetrics(metrics);

        if (outcome === 'success') {
          await this.registerModel();
          this.modelSaveSuccess = true;
          return 'success';
        }

        if (outcome === 'manual') {

          const choice = await this.performManualConfirmation();
          this.resetFlagsForNewStrategy(); // Hide manual UI

          if (choice === 'save') {
            await this.registerModel();
            this.modelSaveSuccess = true;
            return 'success';
          }
          // if choice is 'retrain', the loop will naturally continue to the next strategy
        }

        this.modelSaveError = true;
      }
      return 'failed';
    },

    // Central evaluation logic based on multiple thresholds
    evaluateMetrics(metrics) {
      this.failingMetrics = [];
      let meetsSuccessThreshold = true;
      let meetsManualBaseline = true;

      for (const metricName in this.thresholds) {
        const metricValue = metrics[metricName] || 0;
        const successThreshold = this.thresholds[metricName];

        // Check against success threshold
        if (metricValue * 100 < successThreshold) {
          meetsSuccessThreshold = false;
          this.failingMetrics.push({
            name: metricName,
            value: metricValue,
            threshold: successThreshold
          });
        }

        // Check against hardcoded manual baseline
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
        // 1. Show the test file upload component
        this.testFile = true;

        // 2. Now, create a watcher that waits for the final user choice.
        // This watcher does NOT block the UI from updating.
        const unwatch = this.$watch(() => this.manualChoice, (choice) => {
          if (choice) {
            unwatch(); // Clean up the watcher
            resolve(choice); // Resolve the promise with 'save' or 'retrain'
          }
        });
      });
    },

    // API Call: Standard Training
    async fetchTrain(data) {
      this.loading = true;
      try {
        const response = await axios.post('/autotag/img/fetch_train', data);
        return response.data; // Return the full metrics object
      } catch (error) { console.error('Error in fetchTrain:', error); return undefined; }
      finally { this.loading = false; }
    },
    async trainGanCnn(data) {
      this.loading = true;
      try {
        const response = await axios.post('/autotag/ml/train/fetch_gan_cnn', data);
        return response.data; // Return the full metrics object
      } catch (error) { console.error('Error in GAN-CNN:', error); return undefined; }
      finally { this.loading = false; }
    },

    // API Call: User Data Training
    processUserData(file) {
      this.showUserDataUpload = false;
      this.loading = true;
      this.loadingText = 'Training with User Data...';
      const data = new FormData();
      data.append('tag', this.currentTraining);
      data.append('dataset', file);
      data.append('use_crawled_data', true);

      axios.post('/autotag/img/fetch_train_userdata', data)
        .then(response => {
          if (this.userDataPromiseResolver) {
            this.userDataPromiseResolver(response.data);  // Resolve the promise
          }
        })
        .catch(error => {
          console.error('Error processing user data:', error);
          if (this.userDataPromiseResolver) {
            this.userDataPromiseResolver(undefined); // Resolve with undefined on failure
          }
        })
        .finally(() => {
          this.loading = false;
        });
    },

    // Logic to wait for user data upload
    async userDataInput() {
      this.loading = false;
      this.showUserDataUpload = true;
      return new Promise(resolve => {
        this.userDataPromiseResolver = resolve;
      });
    },

    // API Call: Register Model
    async registerModel() {
      this.loading = true;
      this.loadingText = 'Registering Model...';
      try {
        await axios.post('/autotag/model/register', {
          template: "keras/MultiClassSingleTagKerasStandardModelTemplateA.py",
          group: this.currentTraining,
          model_key: this.currentTraining + '_model.zip',
        });
        console.log("Model Registered!");
      } catch (error) {
        console.error('Error registering model:', error);
      } finally {
        this.loading = false;
      }
    },

    // Resets flags between training keywords or strategies
    resetFlagsForNewKeyword() {
      this.testFile = false;
      this.modelSaveSuccess = false;
      this.modelSaveError = false;
      this.showTestTags = false;
      this.manualChoice = null;
      this.showUserDataUpload = false;
      this.returnedValAccuracy = 0;
    },
    resetFlagsForNewStrategy() {
      this.modelSaveError = false;
      this.modelSaveSuccess = false;
      this.testFile = false;
      this.showTestTags = false;
      this.manualChoice = null;
      this.showUserDataUpload = false;
      this.failingMetrics = [];
    },

    // Resets the entire page to its initial state
    resetPage() {
      Object.assign(this.$data, this.$options.data.apply(this));
    },


    async onTestFileChange(file) {
      this.showTestTags = false;
      this.loading = true;
      this.loadingText = 'Processing Image...';
      const data = new FormData();
      data.append('img', file);
      data.append('temp_tag', this.currentTraining);

      try {
        const response = await axios.post('/autotag/tagTestVerify', data);
        this.showTestTags = true;
        this.test_tags = response.data.tags;

      } catch (error) {
        console.error('Error fetching tags:', error);
        this.showTestTags = false;
        this.test_tags = [];
      }
      finally { this.loading = false; }
    },

    getKeywordStatus(keyword, info) {
      if (info.exists) return 'Model Exists';
      if (keyword === this.currentTraining) return 'Training...';

      const result = this.trainedResults[keyword];
      if (result?.status === 'success') return 'Success';
      if (result?.status === 'failed') return 'Failed';

      return 'Queued';
    },

    getKeywordClass(keyword, info) {
      if (info.exists) return 'status--exists';
      if (keyword === this.currentTraining) return 'status--training';

      const result = this.trainedResults[keyword];
      if (result?.status === 'success') return 'status--success';
      if (result?.status === 'failed') return 'status--failed';

      return 'status--queued';
    }
  }
}
</script>

<style scoped>
/* --- Main Layout & Font --- */
.page-container {
  background-color: #f8f9fa;
  min-height: 100vh;
  padding: 40px 20px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  color: #343a40;
}

.content-wrapper {
  max-width: 800px;
  margin: 0 auto;
}

.animated-fade-in {
  animation: fadeIn 0.6s ease-out;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(-15px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* --- Panels & Headers --- */
.panel {
  width: 100%;
  margin: 20px auto;
  padding: 30px 40px;
  background-color: #ffffff;
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
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
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.btn--primary {
  background-color: #158be3;
  color: white;
}

.btn--primary:hover {
  background-color: #1172bb;
}

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

.btn--secondary:hover {
  background-color: #5a6268;
}

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
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
}

.slider::-moz-range-thumb {
  width: 22px;
  height: 22px;
  background: #158be3;
  border-radius: 50%;
  border: 3px solid white;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
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

.alert--error .alert__title {
  color: #dc3545;
}

.alert--warning {
  background-color: #fff3cd;
  border-color: #ffeeba;
  color: #856404;
}

.alert--warning .alert__title {
  color: #856404;
}

/* --- Status Badges (Used in both components) --- */
.status-badge {
  padding: 5px 12px;
  border-radius: 16px;
  font-size: 13px;
  font-weight: 600;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  line-height: 1.5;
}

.status-badge.status--success {
  background-color: #d1fae5;
  color: #065f46;
}

.status-badge.status--success::before {
  content: '✓';
}

.status-badge.status--failed {
  background-color: #fee2e2;
  color: #991b1b;
}

.status-badge.status--failed::before {
  content: '✗';
}

.status-badge.status--training {
  background-color: #dbeafe;
  color: #1e40af;
}

.status-badge.status--training::before {
  content: '⚙';
  animation: spin 1.5s linear infinite;
}

.status-badge.status--queued {
  background-color: #e5e7eb;
  color: #374151;
}

.status-badge.status--queued::before {
  content: '🕒';
}

.status-badge.status--exists {
  background-color: #e0e7ff;
  color: #3730a3;
}

.status-badge.status--exists::before {
  content: '✓';
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }

  to {
    transform: rotate(360deg);
  }
}

/* ========================================= */
/* == PDF KEYWORD TRAINER SPECIFIC STYLES == */
/* ========================================= */
.keyword-list-section {
  text-align: left;
}

.keyword-list {
  list-style: none;
  padding: 0;
  margin: 0;
  columns: 2;
  gap: 10px;
}

.keyword-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 8px;
  background-color: #f8f9fa;
  margin-bottom: 10px;
  border: 1px solid #e9ecef;
  break-inside: avoid-column;
}

.keyword-item.exists {
  background-color: #f1f3f5;
  color: #adb5bd;
}

.keyword-name {
  font-weight: 600;
  color: #495057;
}

.keyword-score {
  font-size: 13px;
  color: #868e96;
}

.keyword-item .status-badge {
  margin-left: auto;
}

/* --- Training Dashboard --- */

.status-panel {
  flex-grow: 1;
}

/* REMOVED .training-panel as it no longer exists as a separate entity */

.status-grid {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.status-grid__row {
  display: grid;
  grid-template-columns: 1fr 2fr;
  align-items: center;
  background-color: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  overflow: hidden;
  z-index: 1; /* Keep it above the nested panel */
}

.status-grid__cell {
  padding: 15px;
}

.status-grid__cell--keyword {
  font-size: 16px;
  font-weight: 500;
  color: #343a40;
  border-right: 1px solid #e9ecef;
}

.status-grid__cell--status {
  background-color: #fff;
}

/* --- NEW: Nested Training Panel --- */
.training-panel-nested {
  padding: 25px 30px 30px 30px;
  background-color: #ffffff;
  border: 1px solid #e9ecef;
  border-top: none; /* Remove top border as it's connected */
  margin-top: -8px; /* Pull it up to connect with the row above */
  border-radius: 0 0 12px 12px; /* Round only the bottom corners */
  box-shadow: 0 10px 20px rgba(0, 0, 0, 0.03) inset; /* Inner shadow for depth */
}

/* Success Metrics & Progress Bars */
.success-details {
  width: 100%;
}

.metrics-container {
  margin-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.metric-row {
  display: grid;
  grid-template-columns: 80px 1fr 50px;
  align-items: center;
  gap: 10px;
}

.metric-label {
  font-size: 13px;
  color: #495057;
  text-transform: capitalize;
}

.progress-bar-container {
  height: 8px;
  background-color: #e9ecef;
  border-radius: 4px;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background-color: #28a745;
  border-radius: 4px;
  transition: width 0.5s ease-out;
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
</style>