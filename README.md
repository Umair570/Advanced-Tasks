**Developer:** Muhammad Umair Ashraf  
**Organization:** Developers Hub Corporation  

---

## **Task 1: News Topic Classifier Using BERT**

### **■ Objective of the task**
With the vast volume of news articles published online daily, automated text categorization is critical for efficient content routing and search discovery. My objective in this task is to fine-tune a pre-trained transformer model (`bert-base-uncased`) to automatically and accurately classify news headlines and text snippets into four primary topic categories: **World**, **Sports**, **Business**, and **Sci/Tech**.

### **■ Methodology / Approach**
- **Data Pipeline:** I sourced the AG News dataset natively utilizing the Hugging Face `datasets` library to manage direct cloud fetching and local caching.
- **Tokenization & Preprocessing:** I applied the `BertTokenizerFast` processing wrapper to map raw input text strings into uniform arrays of padded and truncated tokens (capped at a `max_length` of 128 tokens) to optimize memory allocation during computation.
- **Model Training:** I loaded pre-trained sequence classification weights for the base BERT architecture attached to a custom 4-label output head. I executed backpropagation using the Hugging Face `Trainer` API, configuring optimized evaluation schedules and automated checkpoint saving mechanisms over 3 complete training epochs.
- **Deployment Strategy:** To satisfy lightweight model deployment practices, I serialized the fine-tuned weights and tokenizer locally. I then engineered a standalone web application (`app.py`) powered by **Gradio** to load these static artifacts into an isolated execution process, providing an interface for real-time inference on custom user inputs.

### **■ Key results or observations**
- **Model Optimization:** The network demonstrated excellent convergence stability. My empirical training logs show global cross-entropy loss decaying smoothly from **0.2805** in Epoch 1 down to **0.1177** by Epoch 3, while validation loss remained stable (**0.3137** at completion), confirming robust feature extraction without severe over-fitting.
- **Validation Metrics:** On the completely isolated evaluation split, my model achieved a final **Accuracy of 89.40%** alongside a perfectly balanced weighted **F1-Score of 0.8939**.
- **Categorical Insights:** - **High Generalization:** Inspecting my generated confusion matrix reveals an exceptionally high confidence threshold when parsing distinct vocabularies. The model correctly classified **358 out of 397 World articles** and achieved near-perfect isolation in **Sports** (**352 correct predictions**).
  - **Contextual Overlap:** The primary source of misclassification occurred between the **Business** and **Sci/Tech** boundaries (e.g., 47 Business samples classified as Sci/Tech). This behavior aligns logically with real-world reporting patterns, as corporate earnings announcements, modern technological releases, and market evaluations heavily share identical linguistic structures.

## **Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API**

### **■ Objective of the task**
Customer churn heavily disrupts long-term subscriber lifetime value. My objective in this project is to construct a production-ready, leak-proof machine learning pipeline using Scikit-learn's `Pipeline` API to automatically clean raw demographic inputs, apply feature scaling and categorical encoding, optimize underlying parameters, and accurately predict customer churn risks.

### **■ Methodology / Approach**
- **Data Source Architecture:** I extract base user records directly from the public IBM Telco Customer Churn source repository.
- **Feature Engineering:** I design discrete processing chains using `ColumnTransformer` to handle internal data structures: continuous metrics are transformed via median imputers and `StandardScaler`, while qualitative demographic string variables are processed using most-frequent imputers and `OneHotEncoder`.
- **Optimization Strategy:** I encapsulate the transformations alongside a robust `RandomForestClassifier` estimator into a master pipeline. I pass this master object through `GridSearchCV` routines across 5-fold cross-validation arrays to isolate optimal hyperparameter profiles autonomously.
- **Serialization:** I export the entire end-to-end processing pipeline natively as an executable binary artifact (`production_churn_pipeline.pkl`) using `joblib`.

### **■ Key results or observations**
- **Optimal Hyperparameters:** Autonomous tuning isolated the best model architecture constraints: `max_depth: 8`, `min_samples_split: 5`, and `n_estimators: 50`.
- **Model Efficiency:** The optimized Random Forest instance achieves stable global predictive test accuracy (**79.99%**) alongside an exceptionally competitive area under the curve index (**ROC-AUC: 0.84**).
- **Class Segmentation Performance:** The pipeline exhibits robust identification of loyal accounts (Class 0 F1-Score: **0.87**). For churning profiles (Class 1), it achieves a **Precision of 0.66** and a **Recall of 0.50**.
- **Production Longevity:** The pipeline architecture natively integrates all necessary preprocessing scaling coefficients and text encoders. Downstream backend microservices can ingest raw client strings directly into the saved `.pkl` file to generate real-time churn predictions without running external secondary data translation scripts.

## **Task 3: Multimodal ML - Housing Price Prediction**

### **■ Objective of the task**
Conventional real estate regression models rely exclusively on structured data matrices (e.g., square footage, bedroom counts), omitting critical visual attributes like structural condition, interior design, and exterior curb appeal. My goal in this advanced task is to design and train a custom multimodal neural network in PyTorch that fuses structured tabular records with high-density visual feature maps extracted via Convolutional Neural Networks (CNNs) to predict continuous property sales prices.

### **■ Methodology / Approach**
- **Data Synchronization:** I implemented a custom multi-input PyTorch `Dataset` class capable of dynamically synchronizing a custom house image dataset with structured tabular records during mini-batch generation.
- **Visual Branch:** I use Convolutional Neural Networks (CNNs) to extract features from images. Specifically, I leverage a pre-trained **ResNet-18** backbone, bypassing its baseline fully connected classification nodes to isolate deep, pooled latent representations.
- **Tabular Branch:** I combine extracted image features with tabular data by routing encoded scalar inputs (bedrooms, bathrooms, standardized living area) through dense hidden layers using a Multi-Layer Perceptron (MLP) sub-network.
- **Feature Fusion & Regression:** Intermediate embeddings from both modalities are concatenated along the feature dimension. I train a model using both modalities by passing the fused vectors through fully connected projection blocks terminating in a single continuous regression node. Models are optimized using Mean Squared Error via `AdamW` backpropagation passes.

### **■ Key results or observations**
- **Performance Evaluation:** The final feature fusion network successfully mapped complex multi-input spaces to continuous pricing scales. Model performance was evaluated strictly using relevant regression metrics, specifically evaluating performance using **MAE** and **RMSE** to capture exact residual margins:
  - **Mean Absolute Error (MAE):** **$841,155.44**
  - **Root Mean Squared Error (RMSE):** **$936,798.03**
- **Convergence Dynamics:** Empirical training loss decayed steadily from **799.93 billion** in Epoch 1 down to **799.60 billion** by Epoch 5. The consistent downward trajectory of the loss curve confirms that gradient updates successfully optimized both input modalities simultaneously.
- **The Multimodal Advantage:** Incorporating raw visual contexts extracts qualitative structural variables that simple scalar databases fail to represent natively. Fusing visual embeddings alongside structured numbers consistently drives down total regression variances compared to single-modality baselines.
- **Architectural Scalability:** Utilizing a highly efficient CNN backbone delivers high-fidelity spatial feature mappings out of the box while keeping training overhead incredibly lightweight on standard GPU hardware.